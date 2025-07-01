# --- Use this if you want to update the helios code ---

# Copy from favyen's dockerfile with flash-attn installed 
FROM gcr.io/ai2-beaker-core/public/d11m0u4pth4i2kqvekq0:latest

# Copy only the updated rslearn_projects code
COPY . /opt/rslearn_projects/

# Reinstall rslearn_projects to include the updates
RUN pip install --no-cache-dir /opt/rslearn_projects

WORKDIR /opt/rslearn_projects 