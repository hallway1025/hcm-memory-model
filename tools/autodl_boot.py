
import paramiko, sys
host, port, user, pwd = os.environ.get("AUTODL_HOST", "<your-host>"), int(os.environ.get("AUTODL_PORT", "22")), os.environ.get("AUTODL_USER", "root"), sys.argv[1]
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, port=port, username=user, password=pwd, timeout=30)
pub = open(sys.argv[2]).read().strip()
cmd = 'mkdir -p ~/.ssh && chmod 700 ~/.ssh && grep -qF "%s" ~/.ssh/authorized_keys 2>/dev/null || echo "%s" >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys' % (pub, pub)
i, o, e = c.exec_command(cmd)
o.channel.recv_exit_status()
i2, o2, e2 = c.exec_command("uname -a; nproc; free -g | head -2; df -h /root/autodl-tmp | tail -1; python --version 2>&1; python -c 'import numpy, torch; print(numpy.__version__, torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))' 2>&1")
print(o2.read().decode("utf-8", "replace"))
c.close()
