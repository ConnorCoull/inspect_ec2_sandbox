"""End-to-end provisioning tests against real EC2 instances.

Each test provisions a sandbox instance via the registered provider and
asserts that a config option took effect on the running instance. Slow
(each instance takes ~2-3 minutes to provision).

Run only with AWS access: ``uv run pytest -m req_aws``. The default
configuration skips them via ``-m "not req_aws"``.
"""

from __future__ import annotations

import pytest

from ec2sandbox._ec2_sandbox_environment import Ec2SandboxEnvironment

from .conftest import (
    build_ec2_config,
    provision,
)

pytestmark = pytest.mark.req_aws


async def test_volume_size_override() -> None:
    # Pick a size that's distinct from any provider default and from the
    # AMI's baked-in size, so the override is observable end-to-end.
    requested_size = 220
    config = build_ec2_config(
        instance_type="t3a.micro",
        volume_size=requested_size,
    )
    envs, _ = await provision(config, "test_volume_size_override")

    try:
        env = envs["default"]
        assert isinstance(env, Ec2SandboxEnvironment)
        # findmnt → root partition, PKNAME → parent disk, /sys/block/<d>/size
        # is in 512-byte sectors. Divide by 2**21 to convert to GiB.
        script = (
            "set -e; "
            "root_part=$(findmnt -no SOURCE /); "
            'disk=$(lsblk -no PKNAME "$root_part"); '
            'sectors=$(cat /sys/block/"$disk"/size); '
            "echo $((sectors / 2097152))"
        )
        result = await env.exec(["bash", "-c", script], timeout=60)
        assert result.success, f"failed to inspect disk size: {result.stderr}"
        observed_gib = int(result.stdout.strip())
        assert observed_gib == requested_size, (
            f"Expected root disk of {requested_size} GiB, observed {observed_gib} "
            f"GiB (stdout={result.stdout!r})"
        )
    finally:
        await Ec2SandboxEnvironment.sample_cleanup(
            task_name="test_volume_size_override",
            config=config,
            environments=envs,
            interrupted=False,
        )
