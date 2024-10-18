# Base image
FROM base-image:latest

# Environment variables
ENV PYTHONPATH="/opt/rslearn_projects:${PYTHONPATH}"
ENV LANDSAT_PORT=5555

# Make port 5555 available to the world outside this container
EXPOSE $LANDSAT_PORT

# Run app.py when the container launches
CMD ["python3", "rslp/landsat_vessels/api_main.py"]