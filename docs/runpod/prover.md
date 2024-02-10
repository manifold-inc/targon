# Running a Prover on RUNPOD

## Introduction
Utilizing Runpod to run a prover offers a streamlined and effective solution for leveraging cloud-based resources. This section details the process of setting up and deploying a prover on Runpod, assuming that you have already created an account and have sufficient funds.

## INSTALLATION

### Create TGI Instance
Manifold has created a template for running TGI on Runpod. You can find the template [here](https://runpod.io/gsc?template=gh1lamgsuz&ref=u908wzaw). Click on the link and you will be redirected to the Runpod website. Once you have logged in, you will be able to create a new instance using the template. The page should look like this once you click the link.

![Runpod TGI Template](runpod_imgs/step_1.png)

Scroll down and seklect A100 as the GPU type. Then click on the "Deploy" button. You will be redirected to the instance page where you can see the status of your instance. It should look like this.

![A100](runpod_imgs/step_2.png)

Then select continue

![Select Continue](runpod_imgs/step_3.png)

then select "Deploy"

![Select Deploy](runpod_imgs/step_4.png)

Once the container is finished building you have set up TGI.

### Set up Prover

Once you have set up TGI, you can now set up the prover. 

Navigate to Pods in the side bar and select + CPU Pod.

![Alt text](runpod_imgs/step_1_prover.png)

Once you have selected the + CPU Pod, you will be redirected to the page where you can set up the CPU pod.

![Alt text](runpod_imgs/step_2_prover.png)

customize the pod to add the prover axon port and then click "Deploy". Once the container is finished building you have set up the prover and follow the readme normally.

![Alt text](runpod_imgs/step_3_prover.png)