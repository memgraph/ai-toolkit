# 🔗 lightrag-memgraph

lightrag-memgraph is an integration that connects
[lightrag](https://github.com/HKUDS/LightRAG) and
[memgraph](https://github.com/memgraph/memgraph). The library began as a small
wrapper designed to specifically configure Memgraph within a pipeline that
processes unstructured data (various texts) and transforms it into an
ontology/entity schema graph. In other words, it enables you to extract and
enhance entities from unstructured documents, storing them in a graph for
powerful querying and analysis. Ideal for building knowledge graphs, improving
data discovery, and leveraging advanced AI techniques on top of your domain
data.

## Notes

- Entity/relationship extraction is high-quality, but also high-cost and
relatively slow.
- The goal over time is to expose time and cost metrics (e.g., $ per your
specific document page or chunk).
