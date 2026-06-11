# Ejecucion con Docker

La imagen incluye Python, las dependencias del proyecto y el modelo
`paraphrase-multilingual-MiniLM-L12-v2`. La maquina que ejecuta la imagen solo
necesita Docker.

## Docker Compose

Construir e iniciar:

```bash
docker compose up --build -d
```

Abrir:

```text
http://localhost:8501
```

Consultar estado y registros:

```bash
docker compose ps
docker compose logs -f
```

Detener:

```bash
docker compose down
```

## Docker sin Compose

Construir la imagen:

```bash
docker build -t demo-cabas-afines:latest .
```

Ejecutar:

```bash
docker run --rm -p 8501:8501 --name demo-cabas-afines demo-cabas-afines:latest
```

## Distribuir sin reconstruir

Exportar la imagen:

```bash
docker save -o demo-cabas-afines.tar demo-cabas-afines:latest
```

En otra maquina, importar y ejecutar:

```bash
docker load -i demo-cabas-afines.tar
docker run --rm -p 8501:8501 demo-cabas-afines:latest
```

La construccion inicial requiere internet para descargar dependencias y el
modelo. La ejecucion posterior de la imagen no requiere acceso a Hugging Face.
