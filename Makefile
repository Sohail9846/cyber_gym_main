# Makefile helpers to run/stop the dev server safely

.PHONY: run run-free stop status logs

run:
	@HOST=0.0.0.0 PORT=$${PORT:-8000} bash scripts/dev_server.sh

# run-free starts at a higher base to avoid common collisions (e.g., 8000)
run-free:
	@HOST=0.0.0.0 PORT=$${PORT:-8100} bash scripts/dev_server.sh

stop:
	@bash scripts/stop_server.sh

status:
	@echo "Active uvicorn processes:" && pgrep -a -f "uvicorn.*main:app" || true; \
	if [ -f devserver.port ]; then echo "Last chosen port:" && cat devserver.port; fi

logs:
	@tail -n 200 -f uvicorn.log
