mkdir -p ~/.streamlit/
echo "
[general]n
email = "maximilien@tutanota.de"n
" > ~/.streamlit/credentials.toml
echo "
[server]n
headless = truen
enableCORS=falsen
port = $PORTn
" > ~/.streamlit/config.toml
[theme]
base="light"
backgroundColor="#f1e4e4"
font="serif"

