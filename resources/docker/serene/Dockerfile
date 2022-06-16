FROM rg.fr-par.scw.cloud/serene/llvm:15-8

ARG TASK=build

WORKDIR /app

COPY . .

RUN ./builder ${TASK}
