module.exports = {
    apps: [
      {
        name: 'ai-moderation',
        script: '/root/neonstream-ai/venv/bin/uvicorn',
        args: 'src.api.main:app --host 127.0.0.1 --port 8000 --workers 2',
        cwd: '/root/neonstream-ai',
        interpreter: 'none',
        env: {
          PYTHONPATH: '/root/neonstream-ai',
          PYTHONUNBUFFERED: '1',
        },
        watch: false,
        autorestart: true,
        max_restarts: 5,
        restart_delay: 3000,
      },
    ],
  };
  