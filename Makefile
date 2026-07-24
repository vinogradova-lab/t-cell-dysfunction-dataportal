.PHONY: sync build dev docker clean

# stage raw inputs from the sibling analysis repo into ./source/
sync:
	python scripts/sync_source.py

# run the ETL: ./source -> ./data/parquet + ./data/downloads
build:
	python etl/build_db.py

# local dev server (after sync + build)
dev:
	python portal/app.py

# production-like stack (runs the ETL inside the image build)
docker: sync
	docker compose up --build

clean:
	rm -rf data/parquet data/downloads source
