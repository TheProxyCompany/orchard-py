from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).parents[2] / ".github" / "workflows").glob("*.yml"))


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda path: path.name)
def test_pie_clone_covers_ssh_submodules_and_keeps_the_token_out_of_argv(
    workflow: Path,
) -> None:
    # PIE lists Carbon as git@github.com:, which an https-only rewrite leaves on
    # the runner's ssh key; and a secret expanded inside a script ends up in a
    # process argument list, readable by every user on the runner.
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    scripts = {
        step.get("name"): step["run"]
        for job in jobs.values()
        for step in job["steps"]
        if "run" in step
    }
    for name, script in scripts.items():
        assert "secrets." not in script, name
        if "proxy-inference-engine.git" in script:
            assert "insteadOf GIT_CONFIG_VALUE_1=git@github.com:" in script, name
