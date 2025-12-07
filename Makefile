.PHONY: setup install clean run

setup:
	@echo "Setting up local environment..."
	@scripts/setup/install_uv.sh
	@uv python install 3.11
	@scripts/setup/create_venv.sh
	@. .venv/bin/activate && make install
	
install:
	@echo "Installing dependencies..."
	uv pip install -e .
	@echo "Generating demo bank database..."
	@psql -U $${PGUSER:-postgres} -h $${PGHOST:-localhost} -p $${PGPORT:-5432} -f scripts/sql/generate_demo_bank.sql || \
		(echo "Error: Failed to create database. Make sure PostgreSQL is running and you have access." && \
		 echo "You may need to set PGHOST, PGPORT, PGUSER, PGPASSWORD environment variables." && \
		 echo "Example: PGUSER=myuser PGPASSWORD=mypass make install" && exit 1)

clean:
	@echo "Cleaning up..."
	rm -rf .venv
	rm -rf dist
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +

run:
	@. .venv/bin/activate
	python -m src.app