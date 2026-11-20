.PHONY: dashboard

dashboard:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_dashboard.ps1
