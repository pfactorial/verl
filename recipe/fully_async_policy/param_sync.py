# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import time

import ray
from ray.util.collective import collective

from verl.utils.device import get_nccl_backend

logger = logging.getLogger(__name__)


@ray.remote
class ParameterSynchronizer:
    """
    Unified parameter synchronizer, responsible for synchronizing model parameters between actor and rollout.

    Supports both vLLM and SGLang backends:
    - vLLM: Uses rollout workers in NCCL collective (direct model access)
    - SGLang: Uses SGLang HTTP server actors in NCCL collective (for optimal performance)
    """

    def __init__(self, config, trainer, rollouter, mq):
        self.config = config
        self.trainer = trainer
        self.rollouter = rollouter
        self.mq_client = mq
        self.actor_wg = ray.get(trainer.get_actor_wg.remote())
        self.rollout_wg = ray.get(rollouter.get_rollout_wg.remote())

        # Detect rollout backend
        self.rollout_name = config.actor_rollout_ref.rollout.name
        self.is_sglang = self.rollout_name == "sglang"

        # For SGLang, get the actual server actors (not ServerAdapter workers)
        self.sglang_servers = None
        if self.is_sglang:
            self.sglang_servers = ray.get(rollouter.get_sglang_servers.remote())
            print(f"[ParameterSynchronizer] Using SGLang backend with {len(self.sglang_servers)} servers")

        # Basic attributes
        self.weights_info = None
        self.sync_group_initialized = False
        self.sync_group_name = "actor_rollout"
        self.wait_last_update = None
        self.wait_last_resume = None

        # Statistics
        self.current_version = 0

        self._init_weights_info()
        self._init_sync_group()

        if self.config.async_training.checkpoint_engine.enable:
            if self.is_sglang:
                # SGLang doesn't use checkpoint_engine (uses NCCL directly to servers)
                print("[ParameterSynchronizer] Skipping checkpoint_engine for SGLang (uses direct NCCL sync)")
            else:
                self._init_actor_rollout_checkpoint_engine()

    def get_current_param_version(self) -> int:
        """Get current parameter version number"""
        return self.current_version

    def get_weights_info(self):
        """Get weights info"""
        return self.weights_info

    def _init_weights_info(self):
        self.weights_info = self.actor_wg.get_actor_weights_info()[0]

        if self.is_sglang:
            # For SGLang, set weights_info on the SGLang servers
            ray.get([server.set_weights_info.remote(self.weights_info) for server in self.sglang_servers])
        else:
            # For vLLM, set on rollout workers
            self.rollout_wg.set_actor_weights_info(self.weights_info)

    def _init_sync_group(self):
        print("[ParameterSynchronizer] Initializing parameter synchronization group...")

        if self.is_sglang:
            # For SGLang, create collective with actor workers + SGLang server actors
            # SGLang servers have GPUs and can participate in NCCL
            actor_sglang_workers = self.actor_wg.workers + self.sglang_servers
            collective.create_collective_group(
                actor_sglang_workers,
                len(actor_sglang_workers),
                list(range(0, len(actor_sglang_workers))),
                backend=get_nccl_backend(),
                group_name=self.sync_group_name,
            )
            print(f"[ParameterSynchronizer] SGLang NCCL group: {len(self.actor_wg.workers)} actors + {len(self.sglang_servers)} SGLang servers")
        else:
            # For vLLM, create collective with actor workers + rollout workers
            actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
            collective.create_collective_group(
                actor_rollout_workers,
                len(actor_rollout_workers),
                list(range(0, len(actor_rollout_workers))),
                backend=get_nccl_backend(),
                group_name=self.sync_group_name,
            )

    def _init_actor_rollout_checkpoint_engine(self):
        ray.get(
            self.actor_wg.init_checkpoint_engine(
                rank_offset=0,
                actor_num=len(self.actor_wg.workers),
                rollout_num=len(self.rollout_wg.workers),
            )
        )
        ray.get(
            self.rollout_wg.init_checkpoint_engine(
                rank_offset=len(self.actor_wg.workers),
                actor_num=len(self.actor_wg.workers),
                rollout_num=len(self.rollout_wg.workers),
            )
        )

    def sync_weights(self, version, validate=False, global_steps=0):
        """Sync weights between trainer and rollouter, and update parameter version"""
        start_time = time.time()

        self.current_version = version
        ray.get(self.rollouter.pause.remote())

        print(f"[ParameterSynchronizer] rollout paused. cost {time.time() - start_time:.2f} seconds")
        # Update MQ version
        self.mq_client.update_param_version_sync(version)

        pause_time = time.time()

        # Sync weights - different paths for SGLang vs vLLM
        if self.is_sglang:
            self._sync_weights_sglang()
        else:
            self._sync_weights_vllm()

        end_time = time.time()
        print(
            f"[ParameterSynchronizer] sync_weights success. cost {end_time - start_time:.2f} seconds, "
            f"pause:{pause_time - start_time:.2f}s, sync:{end_time - pause_time:.2f}s"
        )
        # Async Update rollout version & validation
        self.wait_last_update = self.rollouter.update_param_version.remote(version, validate, global_steps)
        self.wait_last_resume = self.rollouter.resume.remote(self.wait_last_update)

    def _sync_weights_vllm(self):
        """Sync weights for vLLM backend using rollout workers."""
        if self.config.async_training.checkpoint_engine.enable:
            self.actor_wg.sync_rollout_weights_by_checkpoint(self.sync_group_name)
            ray.get(self.rollout_wg.sync_rollout_weights_by_checkpoint(self.sync_group_name))
        else:
            self.actor_wg.sync_rollout_weights(self.sync_group_name)
            ray.get(self.rollout_wg.sync_rollout_weights(self.sync_group_name))

    def _sync_weights_sglang(self):
        """Sync weights for SGLang backend using SGLang server actors.

        This method:
        1. Actor workers broadcast weights via NCCL
        2. SGLang servers receive weights via NCCL and load them into models

        This provides vLLM-equivalent performance because:
        - Uses NCCL for GPU-to-GPU weight transfer (no HTTP serialization)
        - SGLang servers load weights directly
        """
        # Actor workers broadcast weights, SGLang servers receive and load
        # Both happen in parallel via the NCCL collective
        actor_futures = self.actor_wg.sync_rollout_weights(self.sync_group_name)
        sglang_futures = [
            server.sync_weights_from_broadcast.remote(self.sync_group_name)
            for server in self.sglang_servers
        ]

        # Wait for all to complete
        ray.get(actor_futures)
        ray.get(sglang_futures)
        print(f"[ParameterSynchronizer] SGLang weight sync complete: {len(self.sglang_servers)} servers updated")

    def wait_last_valid(self):
        print("[ParameterSynchronizer] Waiting last sync and validate...")
        start_time = time.time()
        if self.wait_last_update:
            ray.get(self.wait_last_update)
        if self.wait_last_resume:
            ray.get(self.wait_last_resume)
        print(f"[ParameterSynchronizer] Wait last validate cost: {time.time() - start_time:.2f} seconds")

    def rollouter_save_checkpoint(self, local_global_step_folder: str):
        """Trigger rollout to save checkpoint(dataloader)"""
        print(f"[ParameterSynchronizer] Triggering checkpoint save at {local_global_step_folder} ...")
        return ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
