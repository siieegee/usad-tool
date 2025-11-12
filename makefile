build:
	sudo docker build -t usad-tool .
run:
	sudo docker compose -p usad-tool up -d
stop:
	sudo docker compose -p usad-tool down