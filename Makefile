.PHONY: setup install index clean run test

# Use the venv interpreter directly so each recipe line (its own shell)
# does not depend on `source .venv/bin/activate` persisting across lines.
PYTHON := .venv/bin/python

setup:
	@echo "Setting up local environment..."
	@scripts/setup/install_uv.sh
	@uv python install 3.11
	@scripts/setup/create_venv.sh
	@$(MAKE) install

install:
	@echo "Installing dependencies..."
	uv pip install -e .
	@echo "Generating demo bank database..."
	@psql -U $${PGUSER:-postgres} -h $${PGHOST:-localhost} -p $${PGPORT:-5432} -f scripts/sql/generate_demo_bank.sql || \
		(echo "Error: Failed to create database. Make sure PostgreSQL is running and you have access." && \
		 echo "You may need to set PGHOST, PGPORT, PGUSER, PGPASSWORD environment variables." && \
		 echo "Example: PGUSER=myuser PGPASSWORD=mypass make install" && exit 1)
	@echo ""
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and fill in your Fireworks API key"
	@echo "  2. Build the schema index: make index"
	@echo "  3. Run the app: make run"

index:
	@echo "Building schema index (requires PostgreSQL + FIREWORKS_API_KEY)..."
	$(PYTHON) -m src.data_prep.build_schema_index

clean:
	@echo "Cleaning up..."
	rm -rf .venv
	rm -rf dist
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +

run:
	$(PYTHON) -m src.app

test:
	$(PYTHON) -m pytest tests/ -v
