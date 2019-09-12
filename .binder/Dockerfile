FROM mcr.microsoft.com/mmlspark/release:latest

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

RUN jupyter-notebook --generate-config \
    && python profile create
USER root
RUN  cp /root/.ipython/profile_default/startup/init_notebook.py /home/jovyan/.ipython/profile_default/startup
USER ${NB_USER}

