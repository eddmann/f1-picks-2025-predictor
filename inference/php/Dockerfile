# PHP with ONNX Runtime for F1 prediction inference
#
# Build:
#   docker build -t f1-picks-2025-predictor-php .
#
# Run:
#   docker run --rm -v $(pwd)/models/saved/onnx:/models f1-picks-2025-predictor-php 2025-24_qualifying.json

FROM php:8.5-cli

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    autoconf \
    g++ \
    make \
    curl \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install ONNX Runtime (must use Microsoft's official release)
# Detect architecture and download appropriate version
ARG ONNX_VERSION=1.22.0
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        ORT_ARCH="linux-aarch64"; \
    else \
        ORT_ARCH="linux-x64"; \
    fi && \
    cd /tmp && \
    curl -L -o onnxruntime.tgz \
    "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-${ORT_ARCH}-${ONNX_VERSION}.tgz" && \
    tar -xzf onnxruntime.tgz && \
    cp -r onnxruntime-${ORT_ARCH}-${ONNX_VERSION}/lib/* /usr/local/lib/ && \
    cp -r onnxruntime-${ORT_ARCH}-${ONNX_VERSION}/include/* /usr/local/include/ && \
    ldconfig && \
    rm -rf /tmp/*

# Build and install PHP ORT extension with ONNX Runtime support
RUN cd /tmp && \
    git clone https://github.com/krakjoe/ort.git && \
    cd ort && \
    phpize && \
    ./configure --with-ort-onnx=/usr/local --enable-ort && \
    make -j$(nproc) && \
    make install && \
    docker-php-ext-enable ort && \
    rm -rf /tmp/*

# Verify ONNX Runtime support is enabled
RUN php --ri ort | grep -q "ONNX Runtime support => enabled" && echo "ONNX Runtime enabled" || (php --ri ort && exit 1)

WORKDIR /app
COPY predict.php ./

ENV MODELS_DIR=/models
ENV LD_LIBRARY_PATH=/usr/local/lib

ENTRYPOINT ["php", "predict.php"]
