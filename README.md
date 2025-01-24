[![DOI](https://zenodo.org/badge/618507646.svg)](https://doi.org/10.5281/zenodo.14735026)

# [Memory-efficient model-based deep learning with convergence and robustness guarantees](https://arxiv.org/pdf/2206.04797.pdf)

Computational imaging has been revolutionized by compressed sensing algorithms, which offer guaranteed uniqueness, convergence, and stability properties. Model-based deep learning methods that combine imaging physics with learned regularization priors have emerged as more powerful alternatives for image recovery. The main focus of this paper is to introduce a memory efficient model-based algorithm with similar theoretical guarantees as CS methods. The proposed iterative algorithm alternates between a gradient descent involving the score function and a conjugate gradient algorithm to encourage data consistency. The score function is modeled as a monotone convolutional neural network. Our analysis shows that the monotone constraint is necessary and sufficient to enforce the uniqueness of the fixed point in arbitrary inverse problems. In addition, it also guarantees the convergence to a fixed point, which is robust to input perturbations. We introduce two implementations of the proposed MOL framework, which differ in the way the monotone property is imposed. The first approach enforces a strict monotone constraint, while the second one relies on an approximation. The guarantees are not valid for the second approach in the strict sense. However, our empirical studies show that the convergence and robustness of both approaches are comparable, while the less constrained approximate implementation offers better performance. The proposed deep equilibrium formulation is significantly more memory efficient than unrolled methods, which allows us to apply it to 3D or 2D+time problems that current unrolled algorithms cannot handle.


## Relevant Paper

A Pramanik, MB Zimmerman, M Jacob, "Memory-efficient model-based deep learning with convergence and robustness guarantees", IEEE Transactions on Computational Imaging, 2023. [IEEE Xplore](https://ieeexplore.ieee.org/document/10059176), [ArXiv version](https://arxiv.org/pdf/2206.04797.pdf)


## Demo Code on Google Colab

A demo code is also provided for reproducibility purposes. Please check it out on [Google Colab](https://colab.research.google.com/drive/1VnMbVW7roOkY_wjpUXUxhNli3BHjwWJB).

## Instructions for Running the Code

* Clone the repository.
* Set the conda environment.
* Carefully, set the parameters in the training script ```trn_mol.py```
* Run the training script using the command: ```python trn_mol.py```
* Once training is finished, perform inference using the testing script ```tst_mol.py```

### Environment

The code has been run on an Nvidia A-100 GPU. The libraries used and their corresponding versions are: 

* Python 3.9
* cudatoolkit 11.3.1
* numpy 1.22
* matplotlib 3.6.2
* scipy 1.7.3
* tqdm 4.64.0
* h5py 3.8.0


## MOL-LR Results for Parallel MRI Recovery

![PMRI](pmri.gif)


## Citation

If you find it useful, please cite the paper as:

```
@article{pramanik2023memory,
  title={Memory-efficient model-based deep learning with convergence and robustness guarantees},
  author={Pramanik, Aniket and Zimmerman, M Bridget and Jacob, Mathews},
  journal={IEEE Transactions on Computational Imaging},
  year={2023},
  publisher={IEEE}
}
```
