FROM python:3.9-buster

# Create arguments to set the default local user information
ARG CONTAINER_USER=ndhuynh
ARG CONTAINER_GROUP=cs-grad
ARG CONTAINER_UID=263606
ARG CONTAINER_GID=3935

# Create a non-root user for the server
RUN groupadd -g ${CONTAINER_GID} ${CONTAINER_GROUP}
RUN useradd -l -ms /bin/bash ${CONTAINER_USER} -u ${CONTAINER_UID} -g ${CONTAINER_GID}

WORKDIR /endure
COPY requirements.txt /tmp

USER root
RUN apt-get update && apt-get install -y vim tmux

USER ${CONTAINER_USER}
ENV PATH "/home/${CONTAINER_USER}/.local/bin:$PATH"
RUN pip install -r /tmp/requirements.txt

EXPOSE 8888

ENTRYPOINT ["bash"]
