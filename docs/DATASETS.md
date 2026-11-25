# Dataset provenance

| Collection and original source | Evaluated input | Role |
|---|---|---|
| [UNSW-NB15, UNSW Canberra](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | 177,906 sampled cleaned rows | Within-domain, transfer, calendar and external training source |
| [Network ToN-IoT, UNSW Canberra](https://research.unsw.edu.au/projects/toniot-datasets) | 179,675 sampled cleaned network rows | Within-domain, transfer, calendar and external training source |
| [CICIoT2023, Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | 195,648 sampled cleaned rows | Within-domain and transfer |
| [RT-IoT2022, UCI](https://archive.ics.uci.edu/dataset/942/rt-iot2022) | 123,117 rows in release order | External target only |

The first three inputs came from cleaned Kaggle mirrors. Parent dataset downloads alone do not identify the evaluated rows. The original export used the mirrors `antonshaubert/unsw-nb15`, `antonshaubert/ton-iot-network` and `himadri07/ciciot2023`. The UNSW cleaned manifest differs from the published parent count by three rows. The cause remains unresolved. [Export metadata](../training_samples/sample_export.json) · [Methods and provenance](../supplement/methods.txt)

Sharmila and Nagapadma's UCI release is identified by DOI 10.24432/C5P338 and distributed under CC BY 4.0. The project retained all original labels for audit and mapped them to attack or benign activity. The observed benign classes are MQTT, ThingSpeak and Wipro bulb. Published estimates use counts from the actual file. [Release record and attribution](../external_data/release.json) · [Observed class counts](../run_results/external_label_counts.csv)

For sample reconstruction, layout and hash checks, see the [data instructions](../data/README.md).
