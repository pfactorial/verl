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

Key features:
- generate_for_partial(): Generation that can be cancelled mid-way
- cancel()/resume(): Cancel and resume rollouts for parameter sync
- load_weights_from_broadcast(): NCCL-based weight sync for optimal performance
"""
import asyncio
import dataclasses
import json
import logging
import os
from typing import Any, Optional, Sequence

import ray
import sglang
import sglang.srt.entrypoints.engine
import torch
from ray.actor import ActorHandle
from sglang.srt.entrypoints.http_server import (
    ServerArgs,
    _GlobalState,
    _launch_subprocesses,
    app,
    set_global_state,
)
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangReplica
from verl.workers.rollout.sglang_rollout.sglang_rollout import _set_envs_and_config
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address, run_unvicorn

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


@ray.remote(num_cpus=1)
class SGLangHttpServerForPartial:
    """SGLang HTTP server with support for partial rollouts (cancellation mid-generation).

    This is a standalone actor class (not inheriting from SGLangHttpServer) that provides:
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
        print(f"SGLang http server (partial): {rollout_mode=}, {replica_rank=}, {node_rank=}, {nnodes=}, {cuda_visible_devices=}")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        assert torch.cuda.is_available(), "SGLang http server should run on GPU node"

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.config.max_model_len = self.config.prompt_length + self.config.response_length
        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.nnodes = nnodes

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for NCCL process group
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address)
            logger.info(
                f"SGLangHttpServerForPartial, replica_rank: {self.replica_rank}, "
                f"master address: {self._master_address}, port: {self._master_port}"
            )
        else:
            self._master_address = None
            self._master_port = None

        # For cancel functionality
        self.paused = False
        self.lock = asyncio.Lock()
        self.cancel_event: dict[str, asyncio.Event] = {}
        self.req_output: dict[str, Optional[dict]] = {}

    def get_master_address(self):
        """Get master address and port for init NCCL process group."""
        return self._master_address, self._master_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port, "non-master node should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
        attention_backend = engine_kwargs.pop("attention_backend", None)
        quantization = self.config.get("quantization", None)
        fp8_block_quant_kwargs = None
        if quantization is not None:
            if quantization == "fp8":
                assert sglang.__version__ >= "0.5.5", "sglang>=0.5.5 is required for FP8 quantization"
                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
            else:
                raise ValueError(f"Currently only support fp8 quantization, got: {quantization}")
        dist_init_addr = (
            f"[{self._master_address}]:{self._master_port}"
            if is_valid_ipv6_address(self._master_address)
            else f"{self._master_address}:{self._master_port}"
        )

        args = {
            "model_path": self.model_config.local_path,
            "dtype": self.config.dtype,
            "mem_fraction_static": self.config.gpu_memory_utilization,
            "disable_cuda_graph": self.config.enforce_eager,
            "enable_memory_saver": True,
            "base_gpu_id": 0,
            "gpu_id_step": 1,
            "tp_size": self.config.tensor_model_parallel_size,
            "dp_size": self.config.data_parallel_size,
            "ep_size": self.config.expert_parallel_size,
            "node_rank": self.node_rank,
            "load_format": self.config.load_format,
            "dist_init_addr": dist_init_addr,
            "nnodes": self.nnodes,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_running_requests": self.config.get("max_num_seqs", None),
            "log_level": "error",
            "mm_attention_backend": "fa3",
            "attention_backend": attention_backend if attention_backend is not None else "fa3",
            "skip_tokenizer_init": self.config.skip_tokenizer_init,
            "skip_server_warmup": True,
            "quantization": quantization,
            "json_model_override_args": json.dumps({"quantization_config": fp8_block_quant_kwargs})
            if quantization == "fp8"
            else json.dumps({}),
            **engine_kwargs,
        }

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

            # start sglang metrics
            args["enable_metrics"] = True

        # enable_weights_cpu_backup is supported in sglang>=0.5.3
        if "enable_weights_cpu_backup" in [f.name for f in dataclasses.fields(ServerArgs)]:
            enable_weights_cpu_backup = True if self.rollout_mode == RolloutMode.COLOCATED else False
            args["enable_weights_cpu_backup"] = enable_weights_cpu_backup

        # NOTE: We can't directly call SGLang's launch_server since it's not an async function.
        # https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py
        sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        server_args = ServerArgs(**args)
        self.tokenizer_manager, self.template_manager, self.scheduler_info, *_ = _launch_subprocesses(
            server_args=server_args
        )

        # In multi-node cases, non-zero rank nodes should not launch http server.
        if self.node_rank > 0:
            return

        set_global_state(
            _GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                template_manager=self.template_manager,
                scheduler_info=self.scheduler_info,
            )
        )
        app.is_single_tokenizer_mode = True

        # Set warmup_thread_args to avoid AttributeError in lifespan function
        app.warmup_thread_args = (
            server_args,
            None,
            None,
        )

        # Manually add Prometheus middleware before starting server
        # This ensures /metrics endpoint is available immediately
        if server_args.enable_metrics:
            from sglang.srt.utils.common import add_prometheus_middleware

            add_prometheus_middleware(app)

        self._server_port, self._server_task = await run_unvicorn(app, server_args, self._server_address)
        self.tokenizer_manager.server_status = ServerStatus.Up

    async def wake_up(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            # Call all workers to switch between trainer mode and rollout mode.
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.resume_memory_occupation(obj, None)
            await self.tokenizer_manager.flush_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.release_memory_occupation(obj, None)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def clear_kv_cache(self):
        obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
        await self.tokenizer_manager.release_memory_occupation(obj, None)

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

    # ==================== Weight Sync Methods ====================

    async def load_weights(self, weights: list[tuple[str, torch.Tensor]], flush_cache: bool = True):
        """Load weights into the SGLang model directly via tokenizer_manager.

        This method receives weights from actor workers (via Ray object store)
        and loads them into the SGLang model by calling tokenizer_manager directly.

        Args:
            weights: List of (name, tensor) tuples to load
            flush_cache: Whether to flush the KV cache after loading (default True)
        """
        from sglang.srt.model_executor.model_runner import LocalSerializedTensor
        from sglang.srt.utils import MultiprocessingSerializer

        # Get inference TP size from config
        infer_tp_size = self.config.tensor_model_parallel_size

        logger.info(f"[SGLang Server {self.replica_rank}] Loading {len(weights)} weight tensors (infer_tp={infer_tp_size})...")

        # Serialize each tensor using SGLang's internal format
        named_tensors = []
        for name, tensor in weights:
            # Ensure tensor is contiguous and detached
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            tensor = tensor.detach()

            # Serialize the tensor for IPC transfer
            serialized_tensor = MultiprocessingSerializer.serialize(tensor)

            # Wrap in LocalSerializedTensor
            # Replicate full weight for all TP ranks - SGLang's model loader handles sharding
            named_tensors.append((name, LocalSerializedTensor(values=[serialized_tensor] * infer_tp_size)))

        # Serialize the entire list of named tensors for each TP rank
        serialized_named_tensors = [
            MultiprocessingSerializer.serialize(named_tensors)
            for _ in range(infer_tp_size)
        ]

        # Create update request and send to tokenizer_manager
        # Pass the request object directly (matches SGLang's HTTP handler behavior)
        req = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_named_tensors,
            load_format=None,
            flush_cache=flush_cache,
        )
        await self.tokenizer_manager.update_weights_from_tensor(req, None)

        logger.info(f"[SGLang Server {self.replica_rank}] Weight loading complete")

    async def flush_cache(self):
        """Flush the KV cache after weight updates."""
        await self.tokenizer_manager.flush_cache()


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
