# Faceted Patient Search API

## Endpoint

```
GET /api/faceted-search/search
```

---

## Description

Faceted, multi-filter patient search powered by MeiliSearch.  
Supports keyword search and filtering by demographics, codes, and encounter statistics.

---

## Query Parameters

| Name             | Type                | Description                                                      | Example                      |
|------------------|---------------------|------------------------------------------------------------------|------------------------------|
| `query`          | string              | Full-text search keywords                                        | `diabetes`                   |
| `gender`         | list of strings     | Filter by gender                                                 | `Female`, `Male`             |
| `age_range`      | list of strings     | Filter by age range                                              | `35-49`, `65+`               |
| `race`           | list of strings     | Filter by race                                                   | `Asian`, `White`             |
| `ethnicity`      | list of strings     | Filter by ethnicity                                              | `Hispanic`                   |
| `insurance_type` | list of strings     | Filter by insurance type                                         | `Medicare`, `Private`        |
| `department`     | list of strings     | Filter by department                                             | `Cardiology`                 |
| `diagnosis_code` | list of strings     | Filter by diagnosis codes                                        | `E11.9`, `R73.03`            |
| `medication_code`| list of strings     | Filter by medication codes                                       | `1552002`, `198145`          |
| `min_encounters` | integer             | Minimum number of encounters                                     | `2`                          |
| `max_encounters` | integer             | Maximum number of encounters                                     | `10`                         |
| `min_age`        | integer             | Minimum age                                                      | `18`                         |
| `max_age`        | integer             | Maximum age                                                      | `65`                         |
| `sort_by`        | string (enum)       | Sort results (`relevance`, `age_asc`, `age_desc`, `encounter_asc`, `encounter_desc`) | `age_desc`                   |
| `limit`          | integer             | Number of results to return                                      | `10`                         |

---

## Request Example

```bash
curl "http://localhost:8000/api/faceted-search/search?gender=Female&min_encounters=2&sort_by=age_desc&limit=5"
```

---

## Response Format

```json
{
  "hits": [
    {
      "patient_id": "124670963",
      "gender": "Male",
      "age": 44,
      "race": "Unknown",
      "ethnicity": "Non-Hispanic",
      "insurance_type": "OTHER (NON-GOVERNMENT)",
      "age_range": "35-49",
      "diagnosis_codes": ["757.39", "E11.9", "..."],
      "medication_codes": ["1552002", "198145", "..."],
      "departments": ["Cardiology"],
      "encounter_count": 12
    },
    ...
  ],
  "facetDistribution": {
    "gender": {"Male": 10, "Female": 15},
    "age_range": {"35-49": 8, "65+": 5},
    ...
  }
}
```

---

## Error Handling

- `400 Bad Request`: Invalid parameter or filter value.
- `500 Internal Server Error`: Unexpected backend or MeiliSearch error.

---

## Notes

- All filter fields are case-sensitive and must match indexed values.
- `query` enables full-text search across searchable attributes.
- Facets are returned for UI filtering and analytics.

---