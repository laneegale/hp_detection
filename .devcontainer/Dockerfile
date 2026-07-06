FROM nvcr.io/nvidia/pytorch:26.06-py3

# Fix the broken third-party repository links that crash standard 'apt' updates
RUN rm -f /etc/apt/sources.list.d/cuda*.list \
    && rm -f /etc/apt/sources.list.d/nvidia*.list \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y libxcb1 libgl1 libglx-mesa0 libglib2.0-0