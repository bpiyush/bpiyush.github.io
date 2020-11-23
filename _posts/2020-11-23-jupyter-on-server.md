---
layout: post
title:  Jupyter on Remote Machine without Port Mapping
date:   2020-11-23
description: How to run and use Jupyter lab on a remote machine without port mapping to your local machine?
---

[Jupyter](https://jupyter.org/) is one of most useful tools for any kind of interactive computing, especially important for research folks who would like to visualise data, models, results and try out bunch of things quickly. This post might be hinted towards a research audience but applies fairly generally to any developer. Often, as researchers, we use remote machines (like AWS instances or on-prem infrastructure). Although Google colab is a brilliant tool of the Jupyter-on-the-cloud kind, it is not feasible to use on our own remote instance since it sits on Google's cloud. If we want to use jupyter on a remote machine, the general practice is to do port-mapping between your local computer and the remote machine. But I find that quite tedious (you have to start a jupyter lab onto the remote machine, do port-mapping locally, and do this every single time). Can we do something like a jupyter-on-the-remote machine accessible through the IP address of the machine? That is precisely what this blog is about.

### Install Jupyter lab

The [official documentation](https://jupyter.org/install) is the best source for installing instructions. I prefer using docker and installing it inside it OR simply in a virtual environment using:

```bash
pip install jupyterlab
```

### Add jupyter config file

After activating the virtual environment, run

```bash
jupyter notebook --generate-config
```

Open the “jupyter_notebook_config.py” configuration file inside the “.jupyter” folder with your preferred text editor. Find the commented out configuration line that defines the value of “c.NotebookApp.ip”, and change the value to ‘0.0.0.0’ to allow remote connections from all IP addresses (realistically, you may not want to do this).

```bash
$ vi ./.jupyter/jupyter_notebook_config.py
# Configuration file for jupyter-notebook.
...
#------------------------------------------------------------------------------
# NotebookApp(JupyterApp) configuration
#------------------------------------------------------------------------------
...
## The IP address the notebook server will listen on.
#c.NotebookApp.ip = 'localhost'
c.NotebookApp.ip = '0.0.0.0'
```

For more information on this, check out this [nice tutorial](https://luppeng.wordpress.com/2017/04/18/remote-access-to-jupyter-notebook/).

### Add a password

Set a login password with the following command:

```bash
$ jupyter notebook password
Enter password:
Verify password:
[NotebookPasswordApp] Wrote hashed password to /home/user/.jupyter/jupyter_notebook_config.json
```

A hash of the password is stored in the file listed above.

### Start Jupyter Lab

```bash
jupyter lab --no-browser --port=8001
```

This would start a jupyter lab at the following address: 

```bash
<Remote-machne-IP>:8001
```

Go to this address, enter the password and that's it! You can keep this running in the backgrounnd using `tmux` or `screen`. You can now work with the Jupyter lab without the worry of broken network etc. If the internet breaks (and this has been run inside `tmux` on the remote machine), the lab will continue to be hosted. You simply need to reload the same address. You can also share this link with your collegues to share results with them! You can work collaboratively on the same lab! I found this to be super useful for my productivity and I hope you find it too! Cheers.

