# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG VLLM_PACKAGE_DIR=/usr/local/lib/python3.12/dist-packages
COPY dspark-pp.patch /opt/dspark/dspark-pp.patch
RUN cd "${VLLM_PACKAGE_DIR}" \
    && patch -p1 --fuzz=0 --dry-run < /opt/dspark/dspark-pp.patch \
    && patch -p1 --fuzz=0 < /opt/dspark/dspark-pp.patch
