Face Verification System
========================

Stack
-----
- Face detection + alignment : MTCNN (facenet-pytorch)
- Embedding model            : InceptionResnetV1 / FaceNet (pretrained VGGFace2)
- Similarity                 : Cosine similarity (sklearn)
- Database                   : MongoDB
- UI                         : Streamlit
- Alerts                     : Gmail (smtplib, built into Python)


Project Structure
-----------------
face_verification_system/
│
├── app.py              Streamlit UI — run this to start the app
├── register.py         Registration logic (folder / webcam / bulk)
├── verify.py           Verification logic (webcam + image upload)
│
├── embedding.py        MTCNN + FaceNet — face detection & embedding
├── similarity.py       Cosine similarity + top-1 matching
├── database.py         MongoDB read/write (users, logs, unknowns)
├── alert.py            Gmail email alerts with image attachment
│
├── config.py           All settings — edit this first
│
├── data/
│   └── processed_data/ Dataset for bulk registration
│       ├── person_1/
│       │   ├── ID_1.jpg
│       │   ├── Selfie_1.jpg
│       │   └── Selfie_2.jpg
│       └── person_2/
│           └── ...
│
├── requirements.txt
└── README.txt


Setup
-----

1. Install dependencies
   pip install -r requirements.txt

   Note: First run downloads FaceNet weights (~90 MB) automatically.

2. Start MongoDB
   Local  : run "mongod" in a terminal
   Cloud  : use MongoDB Atlas free tier (https://www.mongodb.com/atlas)
            paste connection URI into config.py → MONGO_URI

3. Configure config.py
   Open config.py and fill in:
     - MONGO_URI          MongoDB connection string
     - SENDER_EMAIL       Gmail address used to send alerts
     - SENDER_PASSWORD    Gmail App Password (see below)
     - ADMIN_EMAIL        Where alert emails are delivered

4. Gmail App Password (one-time setup)
   - Go to myaccount.google.com
   - Security → enable 2-Step Verification
   - Search "App Passwords" → generate one for Mail
   - Copy the 16-character password into config.py

5. Test email alerts
   python alert.py
   You should receive a test email within seconds.

6. Register faces
   Option A — via Streamlit UI (recommended):
     streamlit run app.py → go to "Register Face" or "Bulk Register"

   Option B — command line:
     python register.py

7. Run the app
   streamlit run app.py


How It Works
------------
Registration:
  Images → MTCNN (detect + align) → FaceNet (512-dim embedding)
  → average all embeddings for a person → store in MongoDB

Verification (webcam or image upload):
  Frame/image → MTCNN → FaceNet → cosine similarity vs all stored embeddings
  → if score >= threshold (default 0.75) → Verified
  → else → Unknown → save snapshot + send alert email

Alert triggers:
  - Unknown face detected           → email with snapshot attached
  - N unknowns in X minutes         → escalated alert email
  - Same person within time window  → duplicate entry alert
  - System error                    → error alert


Configuration Reference (config.py)
------------------------------------
SIMILARITY_THRESHOLD  default 0.75    min cosine score to accept a match
DUPLICATE_WINDOW_SEC  default 60      seconds before same person can re-log
UNKNOWN_ALERT_COUNT   default 3       unknowns before escalated alert fires
UNKNOWN_ALERT_WINDOW  default 5       minutes for the escalation window
DEVICE                default "cpu"   change to "cuda" if GPU available


Notes
-----
- Do NOT push config.py to GitHub if it contains real credentials.
  Add it to .gitignore or replace credentials with placeholders before pushing.
- embeddings are stored as lists in MongoDB (auto-converted to np.array on load).
- Bulk registration skips already-registered persons by default.



Optimizing tool selection...

Read [](file:///d%3A/sem6/CV_project/face_verification_system/app.py#1-1), lines 1 to 50

Read [](file:///d%3A/sem6/CV_project/face_verification_system/requirements.txt)

Read [](file:///d%3A/sem6/CV_project/face_verification_system/README.txt#1-1), lines 1 to 100

Read [](file:///d%3A/sem6/CV_project/face_verification_system/config.py)

## **Face Verification System** — Brief Overview

**Purpose:**  
A real-time face recognition system that registers users' faces and verifies them against a database. Unknown faces trigger email alerts to an admin.

**Core Technologies:**
- **Face Detection & Alignment**: MTCNN (via facenet-pytorch)
- **Face Embedding Model**: InceptionResNetV1 / FaceNet (pretrained on VGGFace2)
- **Similarity Matching**: Cosine similarity (scikit-learn)
- **Database**: MongoDB (stores users, logs, unknowns)
- **UI Framework**: Streamlit (web interface)
- **Alerts**: Gmail (smtplib) for email notifications
- **Languages**: Python with PyTorch, OpenCV

**Workflow:**
1. **Registration** → Images → MTCNN detects & aligns faces → FaceNet generates 512-dimensional embeddings → averaged per user → stored in MongoDB
2. **Verification** → Webcam/image upload → MTCNN + FaceNet embedding → compared against stored embeddings using cosine similarity
3. **Alerts** → Unknown faces (score < 0.75) trigger email notifications with snapshots; escalation if multiple unknowns detected

**Key Features:**
- Bulk registration from dataset folders
- Webcam-based registration/verification
- Real-time video stream verification
- Configurable similarity threshold (default 0.75)
- MongoDB-backed user management
- Automated email alerts with attachments
- Duplicate entry detection within configurable time windows