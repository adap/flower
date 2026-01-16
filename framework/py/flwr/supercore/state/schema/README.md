# State Entity Relationship Diagram

## Schema

```mermaid

---
    config:
        layout: elk
---
erDiagram
  object_children {
    VARCHAR child_id PK,FK
    VARCHAR parent_id PK,FK
  }

  objects {
    VARCHAR object_id PK "nullable"
    BLOB content "nullable"
    INTEGER is_available
    INTEGER ref_count
  }

  run_objects {
    VARCHAR object_id PK,FK
    INTEGER run_id PK
  }

  token_store {
    INTEGER run_id PK "nullable"
    FLOAT active_until "nullable"
    VARCHAR token UK
  }

  objects ||--o| object_children : parent_id
  objects ||--o| object_children : child_id
  objects ||--o| run_objects : object_id

```
