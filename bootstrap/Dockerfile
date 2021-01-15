FROM golang:1-alpine
LABEL maintainer="Sameer Rahmani <lxsameer@gnu.org>"

RUN mkdir -p /usr/src/serene
WORKDIR /usr/src/serene

COPY go.mod go.sum ./
RUN go mod download -x
COPY . .
# Build the Go app
RUN go build -v ./serene.go

CMD ["./serene"]
