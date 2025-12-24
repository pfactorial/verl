# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
SGLang async server with support for partial rollouts (cancellation mid-generation).
This is the SGLang equivalent of recipe/fully_async_policy/vllm_rollout/vllm_async_server.py
"""
import asyncio
import logging
import os
from typing import Any, Optional, Sequence

import ray
from ray.actor import ActorHandle
from sglang.srt.managers.io_struct import GenerateReqInput

from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.sglang_rollout.async_sglang_server import (
    SGLangHttpServer,
    SGLangReplica,
)
from verl.workers.rollout.utils import is_valid_ipv6_address

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


@ray.remote(num_cpus=1)
class SGLangHttpServerForPartial(SGLangHttpServer):
    """SGLang HTTP server with support for partial rollouts (cancellation mid-generation).

    This extends SGLangHttpServer to add:
    - generate_for_partial(): Generation that can be cancelled mid-way
    - cancel(): Cancel all ongoing generations
    - resume(): Resume after cancellation

    These are required for fully async training where the rollouter may need to be
    interrupted when fresh parameters arrive from the trainer.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        super().__init__(
            config, model_config, rollout_mode, workers,
            replica_rank, node_rank, nnodes, cuda_visible_devices
        )

        # For cancel functionality
        self.paused = False
        self.lock = asyncio.Lock()
        self.cancel_event: dict[str, asyncio.Event] = {}
        self.req_output: dict[str, Optional[dict]] = {}

    async def _generate_step(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ):
        """Internal generation step that stores output for later retrieval."""
        max_new_tokens = min(
            self.config.response_length,
            self.config.max_model_len - len(prompt_ids) - 1
        )
        sampling_params = dict(sampling_params)  # Copy to avoid mutating
        sampling_params["max_new_tokens"] = max_new_tokens

        request = GenerateReqInput(
            rid=request_id,
            input_ids=prompt_ids,
            sampling_params=sampling_params,
            return_logprob=True,  # Always need logprobs for training
            image_data=image_data,
        )

        # SGLang uses async generator - get the final output
        output = await self.tokenizer_manager.generate_request(request, None).__anext__()
        self.req_output[request_id] = output

    async def generate_for_partial(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> tuple[Sequence[int], list[float], bool]:
        """
        Generate with ability to cancel mid-generation.

        This is the key method for partial rollouts in fully async training.
        When cancelled, it returns whatever tokens have been generated so far.

        Args:
            prompt_ids: Input token IDs
            sampling_params: Sampling parameters (temperature, top_p, etc.)
            request_id: Unique request identifier for sticky sessions
            image_data: Optional image data for multimodal models

        Returns:
            token_ids: Generated token IDs (may be partial if cancelled)
            log_probs: Log probabilities for each token
            is_cancel: True if generation was cancelled
        """
        async with self.lock:
            if self.paused:
                # After cancel, return immediately with empty results
                return [], [], True

            self.req_output[request_id] = None
            self.cancel_event[request_id] = asyncio.Event()
            cancel_handle = asyncio.create_task(self.cancel_event[request_id].wait())
            generation_handle = asyncio.create_task(
                self._generate_step(prompt_ids, sampling_params, request_id, image_data)
            )

        # Wait for either generation to complete or cancel event
        done, pend = await asyncio.wait(
            [generation_handle, cancel_handle],
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            try:
                await task
            except Exception:
                logger.exception(f"Task failed for request {request_id}")

        for task in pend:
            task.cancel()

        async with self.lock:
            output = self.req_output.get(request_id)
            if output is None:
                return [], [], True

            # Extract token IDs and log probs from SGLang output format
            # SGLang stores logprobs in: output["meta_info"]["output_token_logprobs"]
            # Each entry is a tuple: (log_prob, token_id, ...)
            output_token_logprobs = output.get("meta_info", {}).get("output_token_logprobs", [])

            if output_token_logprobs:
                log_probs, token_ids = zip(
                    *[(lp, tid) for lp, tid, *_ in output_token_logprobs],
                    strict=True
                )
                token_ids = list(token_ids)
                log_probs = list(log_probs)
            else:
                # Fallback if no logprobs (shouldn't happen since we set return_logprob=True)
                token_ids = output.get("output_ids", [])
                log_probs = []

            is_cancel = generation_handle not in done

            # Cleanup
            self.cancel_event.pop(request_id, None)
            self.req_output.pop(request_id, None)

        return token_ids, log_probs, is_cancel

    async def cancel(self):
        """Cancel all ongoing generations.

        Called when the trainer has new parameters and needs to interrupt rollouts.
        """
        async with self.lock:
            self.paused = True
            for request_id in self.cancel_event:
                self.cancel_event[request_id].set()

    async def resume(self):
        """Resume after cancel.

        Called after parameter sync is complete and rollouts can continue.
        """
        async with self.lock:
            self.paused = False


class FullyAsyncSGLangReplica(SGLangReplica):
    """SGLang replica with cancel/resume support for fully async training.

    This extends SGLangReplica to use SGLangHttpServerForPartial instead of
    the regular SGLangHttpServer, enabling mid-generation cancellation.
    """

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)

    async def launch_servers(self):
        """Launch http server in each node.

        This overrides SGLangReplica.launch_servers to use SGLangHttpServerForPartial
        instead of SGLangHttpServer.
        """
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (ray.get_runtime_context().get_node_id(), os.environ["CUDA_VISIBLE_DEVICES"])
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        # create server actor in each node with node affinity and cuda visible devices
        for node_rank in range(self.nnodes):
            workers = self.workers[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            )
            node_id = worker_node_ids[node_rank * self.gpus_per_node]
            name = (
                f"sglang_server_partial_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"sglang_server_partial_reward_{self.replica_rank}_{node_rank}"
            )
            # Use SGLangHttpServerForPartial instead of SGLangHttpServer
            server = SGLangHttpServerForPartial.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

    async def cancel(self):
        """Cancel each rollout server."""
        await asyncio.gather(*[server.cancel.remote() for server in self.servers])

    async def resume(self):
        """Resume each rollout server."""
        await asyncio.gather(*[server.resume.remote() for server in self.servers])
