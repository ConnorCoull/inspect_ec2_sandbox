"""End-to-end cleanup-path tests against real EC2 instances.

Each test provisions one or more sandbox instances via the registered
provider, exercises a specific cleanup path, and asserts the result against
real AWS state. Slow (each instance takes ~2-3 minutes to provision).

Run only with AWS access: ``uv run pytest -m req_aws``. The default
configuration skips them via ``-m "not req_aws"``.
"""

from __future__ import annotations

import time

import boto3
import pytest

from ec2sandbox._ec2_sandbox_environment import (
    MARKER_TAG_KEY,
    Ec2SandboxEnvironment,
    Ec2SandboxEnvironmentConfig,
)

from .conftest import (
    instance_state,
    provision,
    wait_until_terminated,
)

pytestmark = pytest.mark.req_aws

REGION = "eu-west-2"


def _tracked_ids() -> set[str]:
    return {p.instance_id for p in Ec2SandboxEnvironment._tracked_instances}


async def test_happy_path_terminates_via_sample_cleanup(
    ec2_config: Ec2SandboxEnvironmentConfig,
) -> None:
    """sample_cleanup(interrupted=False) terminates the instance and clears the tracker."""  # noqa: E501
    envs, inst_id = await provision(ec2_config, "test_happy")
    try:
        assert inst_id in _tracked_ids()

        await Ec2SandboxEnvironment.sample_cleanup(
            task_name="test_happy",
            config=ec2_config,
            environments=envs,
            interrupted=False,
        )

        assert inst_id not in _tracked_ids()
        assert wait_until_terminated(inst_id) in ("terminated", "shutting-down", None)
    finally:
        # Defensive: even if assertions failed, don't leak the instance.
        if instance_state(inst_id) in ("pending", "running"):
            provider, _ = Ec2SandboxEnvironment._resolve_provider(ec2_config)
            await provider.terminate_instance(inst_id, REGION)


async def test_interrupted_sample_cleanup_skips_then_task_cleanup_sweeps(
    ec2_config: Ec2SandboxEnvironmentConfig,
) -> None:
    """sample_cleanup(interrupted=True) leaves the instance alone; task_cleanup sweeps it."""  # noqa: E501
    envs, inst_id = await provision(ec2_config, "test_interrupted")
    try:
        await Ec2SandboxEnvironment.sample_cleanup(
            task_name="test_interrupted",
            config=ec2_config,
            environments=envs,
            interrupted=True,
        )

        time.sleep(5)
        assert instance_state(inst_id) in ("pending", "running")
        assert inst_id in _tracked_ids()

        # inspect_ai always calls task_cleanup with task_name="shutdown".
        await Ec2SandboxEnvironment.task_cleanup(
            task_name="shutdown", config=ec2_config, cleanup=True
        )

        assert inst_id not in _tracked_ids()
        assert wait_until_terminated(inst_id) in ("terminated", "shutting-down", None)
    finally:
        if instance_state(inst_id) in ("pending", "running"):
            provider, _ = Ec2SandboxEnvironment._resolve_provider(ec2_config)
            await provider.terminate_instance(inst_id, REGION)


async def test_task_cleanup_respects_no_sandbox_cleanup(
    ec2_config: Ec2SandboxEnvironmentConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """task_cleanup(cleanup=False) leaves the instance running but clears the tracker."""  # noqa: E501
    _, inst_id = await provision(ec2_config, "test_no_cleanup")
    try:
        await Ec2SandboxEnvironment.task_cleanup(
            task_name="shutdown", config=ec2_config, cleanup=False
        )

        assert Ec2SandboxEnvironment._tracked_instances == set()
        time.sleep(5)
        assert instance_state(inst_id) in ("pending", "running")
        captured = capsys.readouterr()
        assert "no-sandbox-cleanup" in captured.out.lower()
        assert "inspect sandbox cleanup ec2" in captured.out
    finally:
        provider, _ = Ec2SandboxEnvironment._resolve_provider(ec2_config)
        await provider.terminate_instance(inst_id, REGION)


async def test_multiple_samples_one_interrupted(
    ec2_config: Ec2SandboxEnvironmentConfig,
) -> None:
    """One sample succeeds via sample_cleanup; another, interrupted, is swept by task_cleanup."""  # noqa: E501
    envs_a, inst_a = await provision(ec2_config, "test_multi")
    envs_b, inst_b = await provision(ec2_config, "test_multi")
    try:
        assert {inst_a, inst_b}.issubset(_tracked_ids())

        # Sample A succeeds.
        await Ec2SandboxEnvironment.sample_cleanup(
            task_name="test_multi",
            config=ec2_config,
            environments=envs_a,
            interrupted=False,
        )
        assert inst_a not in _tracked_ids()
        assert inst_b in _tracked_ids()

        # Sample B interrupted.
        await Ec2SandboxEnvironment.sample_cleanup(
            task_name="test_multi",
            config=ec2_config,
            environments=envs_b,
            interrupted=True,
        )
        assert inst_b in _tracked_ids()

        # task_cleanup sweeps the leftover.
        await Ec2SandboxEnvironment.task_cleanup(
            task_name="shutdown", config=ec2_config, cleanup=True
        )

        assert inst_b not in _tracked_ids()
        assert wait_until_terminated(inst_a) in ("terminated", "shutting-down", None)
        assert wait_until_terminated(inst_b) in ("terminated", "shutting-down", None)
    finally:
        provider, _ = Ec2SandboxEnvironment._resolve_provider(ec2_config)
        for inst in (inst_a, inst_b):
            if instance_state(inst) in ("pending", "running"):
                await provider.terminate_instance(inst, REGION)


async def test_cli_cleanup(ec2_config: Ec2SandboxEnvironmentConfig) -> None:
    _, new_instance_id = await provision(ec2_config, "test_cli_cleanup")

    await Ec2SandboxEnvironment.cli_cleanup(id=None)

    post_cleanup_instance_ids = _read_instance_ids()

    assert new_instance_id not in post_cleanup_instance_ids


def _read_instance_ids() -> set[str]:
    ec2 = boto3.client("ec2")
    response = ec2.describe_instances(
        Filters=[
            {
                "Name": f"tag:{MARKER_TAG_KEY}",
                "Values": ["true"],
            },
            {
                "Name": "instance-state-name",
                "Values": ["pending", "running", "stopping", "stopped"],
            },
        ]
    )
    instance_ids = set()
    for reservation in response["Reservations"]:
        for instance in reservation["Instances"]:
            instance_ids.add(instance["InstanceId"])

    return instance_ids
