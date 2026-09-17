# Data dictionary (Qualtrics item IDs)

This is a working dictionary reconstructed from the survey exports and `NbS_TUSSIE.R`. It is not the official Qualtrics codebook.

## Identifiers and survey metadata

| Field | Description |
| --- | --- |
| `ResponseId` | Qualtrics response ID; merge key |
| `Finished` | Whether the respondent reached the end of the survey |
| `StartDate`, `EndDate`, `RecordedDate` | Timestamps in the raw CSV |
| `Progress`, `Duration (in seconds)` | Survey progress metadata in the raw CSV |

## Section 1. Respondent and farm

| Item | Description |
| --- | --- |
| `q1.1` | UK nation/region |
| `q1.2` | Age group |
| `q1.3` | Gender (`Man` / `Woman`; other text in `q1.3_3_TEXT`) |
| `q1.4` | Highest educational attainment |
| `q1.5` | Annual household income after tax |
| `q1.6` | Farm area |
| `q1.7_*` | Role/tenure: landowner, tenant, worker, family member, other |
| `q1.8_*` | Farm activities (cereals, horticulture, dairy, grazing, pigs, poultry, mixed, other) |
| `q1.9` | Share of household income from activities other than farming |
| `q1.10_*` | Funding sources (public schemes, private, other, none) |

## Section 2. Extreme weather and climate risk

| Item | Description |
| --- | --- |
| `q2.1` | Extreme weather / pest / disease events experienced |
| `q2.2` | Frequency of events |
| `q2.3` | How strongly the farm was affected |
| `q2.4` | Attribution of events to climate change |
| `q2.5_1` to `q2.5_4` | Perceived climate risk to present/future yield and farm business |

## Section 3. Collaboration

| Item | Description |
| --- | --- |
| `q3.1` | Membership of a farmer cluster / partnership |
| `q3.2.1_*` / `q3.2.2_*` | Attitudes to collaborating with other farmers |

## Section 4. Nature-based Solutions

| Item | Description |
| --- | --- |
| `q4.1` | Implements NbS: no / individually / with other farmers |
| `q4.2` | Types of NbS practices |
| `q4.3` | Support received for NbS |
| `q4.4` | Whether practices have changed |
| `q4.5` | Whether neighbours implement NbS |
| `q4.6.*` | Perceptions of costs, protection, productivity, time horizon, land trade-offs, collaboration |

## Section 5. Governance and challenges

| Item | Description |
| --- | --- |
| `q5.1_*` | Views on government promotion of NbS and related statements |
| `q5.2_*` | Self-reported challenges (implementation costs, maintenance, policy uncertainty, tenancy, etc.) |
| `q5.3` | Final open comments / additional challenges |

## Files

- `Results_text.xlsx`: labelled answers.
- `Results_sep.xlsx`: dummy-style split codes for multi-select items.
- `Results_values.xlsx`: numeric codes.
- CSV export: full Qualtrics dump with two header/metadata rows.

Open-text `_TEXT` columns contain free comments. Review them before making the repository public.
