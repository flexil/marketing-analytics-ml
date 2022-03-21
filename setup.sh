mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[theme]\n\
base = 'light'\n\
\n\
backgroundColor = '#f1e4e4'\n\
\n\
font='serif'\n\
\n\
" > ~/.streamlit/config.toml
