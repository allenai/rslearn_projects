## Tolbi Project

This project focuses on mapping a variety of cash crops (palm oil, cocoa, rubber, etc) in the Ivory Coast region. The Tolbi team only sent us positive samples, and we need to extract negative samples to ensure the model can correctly identify all the cash crops.

### Extract Negative Samples from WorldCover

WorldCover 16.5M labels: `/weka/dfive-default/rslearn-eai/artifacts/WorldCover/final_reference_data.csv`

The geometry for the Ivory Coast is:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [
              -8.575678982928736,
              10.525361785272338
            ],
            [
              -8.575678982928736,
              4.61955154569263
            ],
            [
              -2.665272278762558,
              4.61955154569263
            ],
            [
              -2.665272278762558,
              10.525361785272338
            ],
            [
              -8.575678982928736,
              10.525361785272338
            ]
          ]
        ],
        "type": "Polygon"
      }
    }
  ]
}
```

['grassland' 'shrub' 'tree' 'wetland (herbaceous)' 'crops'
 'fallow/shifting cultivation' 'bare' 'Not sure' 'urban/built-up' 'burnt'
 'water']