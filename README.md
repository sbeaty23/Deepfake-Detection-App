# Deepfake Detection App

Senior capstone project by Shawn Beaty, Abdullah Al-Quaid, Patrick Scheib, and JP DiGuiseppe.

The initial ML research focuses on identifying fully AI-generated portraits, while documenting limitations, false-positive risks, and generalization to unfamiliar generators. Components are under active development.

## Repository guide

| Folder | Contents |
| --- | --- |
| [`frontend/`](frontend/) | React / TypeScript user interface |
| [`backend/`](backend/) | Backend API and integration (in development) |
| [`ml/`](ml/) | PyTorch model experiments and evaluation |
| [`forensics/`](forensics/) | Dataset manifests, deduplication, grouping, and split auditing |
| [`docs/ml/`](docs/ml/) | ML research, protocol, and experiment documentation |
| [`docs/forensics/`](docs/forensics/) | Forensics and dataset research documentation |

## Project plan

[Team project plan (SharePoint)](https://wumail-my.sharepoint.com/:w:/r/personal/srbeaty_widener_edu/Documents/Deepfake_Detection_Comprehensive_Project_Plan.docx?d=w0acee46458f04620b7ec43bb7b2e40c9&csf=1&web=1&e=fVG9iu)

**Development workflow:** Work on the relevant branch, then open a pull request for team review before merging to `main`. The `docs/ml` and `docs/forensics` names are branches as well as documentation folder paths; branches and folders are distinct Git concepts.
