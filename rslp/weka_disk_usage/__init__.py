"""Weka disk usage scanner, collapsed-tree viewer/report, and weekly Beaker job.

See README.md in this directory for the scan -> collapse -> view/report pipeline and
for the weekly report that is launched from the GitHub Action in
.github/workflows/weka_disk_usage.yaml.
"""

from .launch_weekly_job import launch_weekly_job
from .weekly_report import weekly_report

workflows = {
    "weekly_report": weekly_report,
    "launch_weekly_job": launch_weekly_job,
}
