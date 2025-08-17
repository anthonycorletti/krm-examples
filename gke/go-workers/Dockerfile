FROM golang:1.23-alpine

WORKDIR /app

COPY ./go-workers/ /app/

RUN go build -o github-scraper main.go

EXPOSE 8080

CMD ["./github-scraper"]
