# Neural Style Transfer on a Jetson TX1

## Flashing the Jetson
The Jetson comes with some basic software installed (Ubuntu, CUDA, drivers, etc.).  
However, the setup process on the Jetsono can be a bit whimsical and easy to get stuck in. 
It's possible that somewhere along the line you (like me) will screw up the Jetson to where it must be re-imaged.
Thus, I'll start with the process of 'flashing' the Jetson from a host machine.
You can skip this step to start out, but its useful to master in case you ever get stuck.

To flash the Jetson, you must have a computer which is running Ubuntu 14.04 (it is possible with Ubuntu 16, but you may run into difficulties).
Since I run Windows, this will require using a virtual machine. 
I followed the instructions [found here](https://medium.com/@connerfritz/installing-the-latest-jetpack-on-a-nvidia-jetson-tx1-on-windows-through-virtualbox-8adef92e7171)
to a tee, with one exception:   

When you download an Ubuntu iso, don't do it from the Ubuntu Desktop site as they give you the latest version (18). 
Instead, download the iso image from [here](http://releases.ubuntu.com/14.04/) to get Ubuntu 14.04. 
Another tip is to make sure you have at least 30G of disk space clear on your main Windows machine before you start this process otherwise you will run out of space on your VM.  

In the tutorial, they only have you flash the OS, we will also need CUDA toolkit as it is a prereq for pytorch.
Another advantage of doing this process is that we won't have all the unneeded additional software that comes with jetpack and we can save a little space (the Jetson doesn't have much free disk space).

## Set up memory swap
We are going to run out of CPU trying to install the necessary python packages. 
Thus, we will need to create a swapfile to allow the Jetson to free up space.
Plug a usb drive (at least 8G, I used 32G) into the Jetson. 
Then run the code below to get and run a useful script for making a swapfile.  

```console
git clone https://github.com/jetsonhacks/postFlashTX1
cd postFlashTX1
sudo ./createSwapfile.sh -d '/media/ubuntu/[name of your usb]' -s 10 -a
```  

Now follow the instructions in this [video walkthrough](https://www.youtube.com/watch?v=BB5AVnBQNo4) to take care of configurations for mounting the usb on startup.
Finally, restart the Jetson.  

## Dependencies
```console
# Get python3.6
sudo apt-get update
sudo apt-get install python3.6

# Get pip3
sudo apt-get install software-properties-common
sudo apt-add-repository universe
sudo apt-get update
sudo apt-get install python-pip
```

## Installing PyTorch
For this project, I wanted to use pytorch as it is the library I've been focusing my 'deep' learning arounding (I got jokes).
To this end, I followed [these instructions](https://gist.github.com/dusty-nv/ef2b372301c00c0a9d3203e42fd83426) to install pytorch from source.
The reason we have to do this in an atypical fashion is that the Jetson is running aarch64 and isn't compatable with standard package manager installation (i.e. pip or conda).  
