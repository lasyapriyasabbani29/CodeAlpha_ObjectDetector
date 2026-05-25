STREAMLIT_CSS = """
<style>
:root {
  --bg: #070a0f;
  --panel: #0e141d;
  --panel-soft: #121b25;
  --line: #273244;
  --text: #edf3f8;
  --muted: #93a4b7;
  --cyan: #20d7ff;
  --green: #3de09d;
  --amber: #ffb24a;
  --rose: #ff5a7a;
}

.stApp {
  background:
    radial-gradient(circle at 18% 18%, rgba(32, 215, 255, 0.09), transparent 30%),
    radial-gradient(circle at 80% 0%, rgba(61, 224, 157, 0.08), transparent 28%),
    var(--bg);
  color: var(--text);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #091018 0%, #0d131c 100%);
  border-right: 1px solid var(--line);
}

[data-testid="stMetric"] {
  background: rgba(14, 20, 29, 0.88);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
}

.status-card {
  background: rgba(14, 20, 29, 0.84);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
}

.status-card strong {
  color: var(--cyan);
}

.event-row {
  border-left: 3px solid var(--amber);
  padding: 7px 10px;
  margin: 6px 0;
  background: rgba(255, 178, 74, 0.08);
  border-radius: 6px;
  color: var(--text);
}

h1, h2, h3 {
  letter-spacing: 0;
}

button[kind="primary"] {
  border: 1px solid rgba(32, 215, 255, 0.55);
}
</style>
"""

