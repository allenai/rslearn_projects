# Base image
FROM base-image:latest

# Install tippecanoe.
RUN git clone https://github.com/mapbox/tippecanoe /opt/tippecanoe
WORKDIR /opt/tippecanoe
RUN make -j
RUN make install

WORKDIR /opt/rslearn_projects
