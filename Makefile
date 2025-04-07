PROJECT_NAME = chat-bot

# Build the Docker image
build:
	docker build --no-cache -t $(PROJECT_NAME) .

# Run the Docker container
run:
	./docker_run.sh

# Stop all running containers
stop:
	docker stop $(PROJECT_NAME)

# Clean up unused containers and images
clean:
	docker system prune -f

# Remove the Docker image
remove-image:
	docker rmi $(PROJECT_NAME) || true

# Rebuild the Docker image (clean, remove image, and build)
rebuild:
	make clean
	make remove-image
	make build --no-cache -t $(PROJECT_NAME) .

# Show all containers (including stopped ones)
status:
	docker ps -a

# Run tests using pytest
test:
	docker run --rm $(PROJECT_NAME) pytest tests/ 

# Build, run tests, and remove the container
test-run:
	make build
	make test
	make stop
