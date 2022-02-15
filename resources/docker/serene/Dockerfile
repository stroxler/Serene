FROM rg.fr-par.scw.cloud/serene/llvm:15-4

ARG TASK=build

WORKDIR /app

COPY . .

RUN ./builder ${TASK}
