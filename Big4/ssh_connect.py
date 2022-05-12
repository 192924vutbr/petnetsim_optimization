import paramiko
import sys


server = "uai-gpu.fme.vutbr.cz"
login = "s192924"
passwd = "3g5pHC#uxVnWQ3UH"

results = []
def connect():
    client =  paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect(server, username= login, password=passwd)

    # Setup sftp connection and transmit this script
    sftp = client.open_sftp()
    sftp.put("big4_ssh.py", '/tmp/big4_ssh.py')
    sftp.close()

    ssh_stdin, ssh_stdout, ssh_stderr = client.exec_command("ls -1 /tmp")

    for line in ssh_stdout:
        results.append(line.strip('\n'))

connect()
for i in results:
    print((i.strip()))

#sys.exit()