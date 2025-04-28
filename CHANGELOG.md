## 1.0.0 (2025-04-28)

### Features

* optimize confidence score function ([d1df20b](https://github.com/w-disaster/transcendent-multiclass/commit/d1df20bf5a67b9d8ba9c59f20e29f9f8cc335b7a))
* optimize random forest proximities using parallel processing ([d428ba5](https://github.com/w-disaster/transcendent-multiclass/commit/d428ba5362d9ffd0a54876768f93315199a13999))
* parallel RF proximities ([7a26b6a](https://github.com/w-disaster/transcendent-multiclass/commit/7a26b6a6e66db31e07dde0bdddc69550ee98c4bf))

### Bug Fixes

* change version ([3e88471](https://github.com/w-disaster/transcendent-multiclass/commit/3e88471611e927088f5f1fdc8eb0953a1298df16))
* credibility score ([23c574f](https://github.com/w-disaster/transcendent-multiclass/commit/23c574fa085af9878d83b066ec294ee9057419a8))
* family mapping ([c8cdc1f](https://github.com/w-disaster/transcendent-multiclass/commit/c8cdc1f21a76f9d6b99cffad7b400872cd6ab011))

### Build and continuous integration

* add Dockerfile, trigger image deploy on main branch ([943a9bc](https://github.com/w-disaster/transcendent-multiclass/commit/943a9bcf0d2a0d6e8e917b6dca45b2fcbf9c0c53))
* delete coverage tests ([eb1550b](https://github.com/w-disaster/transcendent-multiclass/commit/eb1550b522af5e911b34f255fe2771fca9708622))
* remove change logging level ([d0d8b78](https://github.com/w-disaster/transcendent-multiclass/commit/d0d8b7835a62a8f3a3207f4175499a64917c17c3))
* restore coverage ([cbe2d85](https://github.com/w-disaster/transcendent-multiclass/commit/cbe2d85abeeb754524ccc98f97ea1faf159a8595))

### General maintenance

* add concept drift notebook, update confidence function for multiclass ([d6de3cd](https://github.com/w-disaster/transcendent-multiclass/commit/d6de3cd61b9a1ff0a1cae6d9e1beb09ee672cd58))
* add shared memory utils ([ed09def](https://github.com/w-disaster/transcendent-multiclass/commit/ed09def985763a3ac1e4f16a6eca5d6ce7929a27))
* add shm to ret ([08f5b54](https://github.com/w-disaster/transcendent-multiclass/commit/08f5b5479dd0e26615fc7f6660bd49c83b083713))
* add stats of incorrect testing samples with high cred ([f7519d4](https://github.com/w-disaster/transcendent-multiclass/commit/f7519d40de29f9afdf720e9827570c02b39a4488))
* add template files ([89a879c](https://github.com/w-disaster/transcendent-multiclass/commit/89a879c10cae2eb6bce4b98486768a8b899d38b3))
* delete try finally and temporary disable shm close/unlink methods ([28c7942](https://github.com/w-disaster/transcendent-multiclass/commit/28c794229170a68b2019c59a50e2227f6860addf))
* format code using ruff ([92a2801](https://github.com/w-disaster/transcendent-multiclass/commit/92a2801cb1b8ecdfd8e7d0c58d50a8f685904d80))
* format code, remove pycache dirs ([98f4cc7](https://github.com/w-disaster/transcendent-multiclass/commit/98f4cc75293f11bd6325542dbf0157734765e6fc))
* map families to int ([f9f8acb](https://github.com/w-disaster/transcendent-multiclass/commit/f9f8acb73137674e26781ed96ac358549f449671))
* **release:** 1.0.0 [skip ci] ([849cb22](https://github.com/w-disaster/transcendent-multiclass/commit/849cb22efcf2a7d575cc2f9441fe5702cb171149))
* **release:** 1.0.0 [skip ci] ([b1dff26](https://github.com/w-disaster/transcendent-multiclass/commit/b1dff26d7350096fb50e53e46585aa90a552b7f0))
* **release:** 1.0.0 [skip ci] ([99e9112](https://github.com/w-disaster/transcendent-multiclass/commit/99e91127ac704ba1d76e7bcd6958370ed5af04ef))
* **release:** 1.0.0 [skip ci] ([c519bb0](https://github.com/w-disaster/transcendent-multiclass/commit/c519bb057b5219cfb7130ad5df95e8a22a3dab67))
* remove tqdm in rf ncm ([efe466d](https://github.com/w-disaster/transcendent-multiclass/commit/efe466d152e3e873dfcd47beb08e6f75b84c002f))
* remove unused files ([0342fc7](https://github.com/w-disaster/transcendent-multiclass/commit/0342fc798de586e04c1504b8c78b89245194d33a))
* remove useless files ([28b04ca](https://github.com/w-disaster/transcendent-multiclass/commit/28b04cada7f13a92216a32f10bb837ffb924318d))
* remove useless files, update gitignore ([7055fe1](https://github.com/w-disaster/transcendent-multiclass/commit/7055fe1c9d30e98ceeda38c715190b43dd624fcd))
* rename package to transcendent ([471cd0f](https://github.com/w-disaster/transcendent-multiclass/commit/471cd0f3c650b18cbe580e024655e977144d8d17))
* rename project ([c87fc24](https://github.com/w-disaster/transcendent-multiclass/commit/c87fc240c72d79f796142519f2ac4a8188eeb8bb))
* restore malware dataset ([c1e29eb](https://github.com/w-disaster/transcendent-multiclass/commit/c1e29eb69dbe7c5f64898d6a5fb2e67bae68bfe3))
* restore run script ([dbfac36](https://github.com/w-disaster/transcendent-multiclass/commit/dbfac36d6ff5b1945a053f480932002036b059f6))
* rf-10-knn to rf-5-knn ([09031b7](https://github.com/w-disaster/transcendent-multiclass/commit/09031b7944c7f827c280a3cc5f27559f13b81d38))
* rm changelog ([787df4d](https://github.com/w-disaster/transcendent-multiclass/commit/787df4db92e90872d983ad3e98b71964f144fc7b))
* rm Changelog ([33637ea](https://github.com/w-disaster/transcendent-multiclass/commit/33637ea07382737409fc5ce1c1bb2fbb3099136c))
* update concept drift notebook ([39ea4f8](https://github.com/w-disaster/transcendent-multiclass/commit/39ea4f8a0f0965ad88a20799aa256c84552155ea))
* update dataset loading ([9d6d5a3](https://github.com/w-disaster/transcendent-multiclass/commit/9d6d5a39c4567e04f6ba8f5a80e43af4d80cd7f7))
* update gitignore ([707d58d](https://github.com/w-disaster/transcendent-multiclass/commit/707d58d96de4cbadd463163db6166997ea518d50))
* update notebook with correct conf score ([c48a1d9](https://github.com/w-disaster/transcendent-multiclass/commit/c48a1d98b6d303ec0b564de6081700d39c0397b4))
* use all labels in ice for testing stats ([f7f35ee](https://github.com/w-disaster/transcendent-multiclass/commit/f7f35eeff186f7d64a3416948ce6405424641486))
* use malware dataset ([5c5b42d](https://github.com/w-disaster/transcendent-multiclass/commit/5c5b42d2579c39eb1e45e3ad3679686ce398e26b))
* use micro as scores average ([561e0aa](https://github.com/w-disaster/transcendent-multiclass/commit/561e0aa3cddaa3809448e11bdf0d038c4f6b0c2d))
* use random forest classifier instead of svm, ncm based on rf proximities ([82a4b16](https://github.com/w-disaster/transcendent-multiclass/commit/82a4b166b5bfe90158e27b32aeb5766afe228511))

### Style improvements

* format code ([84bdafd](https://github.com/w-disaster/transcendent-multiclass/commit/84bdafdb9e4109aeb9968af0f2fd53dfef4fa706))
* format code ([bb9433b](https://github.com/w-disaster/transcendent-multiclass/commit/bb9433bf67888fbd524e06525ea043604fba376e))

### Refactoring

* remove data split analysis dir ([a243502](https://github.com/w-disaster/transcendent-multiclass/commit/a24350251c883363473c0da678e5e27f050adada))
* remove useless file ([70c28df](https://github.com/w-disaster/transcendent-multiclass/commit/70c28dfbd1cb8cbd7e322025afcf737f7121bbc8))
* remove useless files ([ad1c78a](https://github.com/w-disaster/transcendent-multiclass/commit/ad1c78ae07cfddf8fcc26444a0996757bc4805bd))

## 1.0.0 (2025-04-28)

### Features

* optimize confidence score function ([d1df20b](https://github.com/w-disaster/transcendent-multiclass/commit/d1df20bf5a67b9d8ba9c59f20e29f9f8cc335b7a))
* optimize random forest proximities using parallel processing ([d428ba5](https://github.com/w-disaster/transcendent-multiclass/commit/d428ba5362d9ffd0a54876768f93315199a13999))
* parallel RF proximities ([7a26b6a](https://github.com/w-disaster/transcendent-multiclass/commit/7a26b6a6e66db31e07dde0bdddc69550ee98c4bf))

### Bug Fixes

* change version ([3e88471](https://github.com/w-disaster/transcendent-multiclass/commit/3e88471611e927088f5f1fdc8eb0953a1298df16))
* credibility score ([23c574f](https://github.com/w-disaster/transcendent-multiclass/commit/23c574fa085af9878d83b066ec294ee9057419a8))
* family mapping ([c8cdc1f](https://github.com/w-disaster/transcendent-multiclass/commit/c8cdc1f21a76f9d6b99cffad7b400872cd6ab011))

### Build and continuous integration

* add Dockerfile, trigger image deploy on main branch ([943a9bc](https://github.com/w-disaster/transcendent-multiclass/commit/943a9bcf0d2a0d6e8e917b6dca45b2fcbf9c0c53))
* delete coverage tests ([eb1550b](https://github.com/w-disaster/transcendent-multiclass/commit/eb1550b522af5e911b34f255fe2771fca9708622))
* remove change logging level ([d0d8b78](https://github.com/w-disaster/transcendent-multiclass/commit/d0d8b7835a62a8f3a3207f4175499a64917c17c3))
* restore coverage ([cbe2d85](https://github.com/w-disaster/transcendent-multiclass/commit/cbe2d85abeeb754524ccc98f97ea1faf159a8595))

### General maintenance

* add concept drift notebook, update confidence function for multiclass ([d6de3cd](https://github.com/w-disaster/transcendent-multiclass/commit/d6de3cd61b9a1ff0a1cae6d9e1beb09ee672cd58))
* add shared memory utils ([ed09def](https://github.com/w-disaster/transcendent-multiclass/commit/ed09def985763a3ac1e4f16a6eca5d6ce7929a27))
* add shm to ret ([08f5b54](https://github.com/w-disaster/transcendent-multiclass/commit/08f5b5479dd0e26615fc7f6660bd49c83b083713))
* add stats of incorrect testing samples with high cred ([f7519d4](https://github.com/w-disaster/transcendent-multiclass/commit/f7519d40de29f9afdf720e9827570c02b39a4488))
* add template files ([89a879c](https://github.com/w-disaster/transcendent-multiclass/commit/89a879c10cae2eb6bce4b98486768a8b899d38b3))
* delete try finally and temporary disable shm close/unlink methods ([28c7942](https://github.com/w-disaster/transcendent-multiclass/commit/28c794229170a68b2019c59a50e2227f6860addf))
* format code using ruff ([92a2801](https://github.com/w-disaster/transcendent-multiclass/commit/92a2801cb1b8ecdfd8e7d0c58d50a8f685904d80))
* format code, remove pycache dirs ([98f4cc7](https://github.com/w-disaster/transcendent-multiclass/commit/98f4cc75293f11bd6325542dbf0157734765e6fc))
* map families to int ([f9f8acb](https://github.com/w-disaster/transcendent-multiclass/commit/f9f8acb73137674e26781ed96ac358549f449671))
* **release:** 1.0.0 [skip ci] ([b1dff26](https://github.com/w-disaster/transcendent-multiclass/commit/b1dff26d7350096fb50e53e46585aa90a552b7f0))
* **release:** 1.0.0 [skip ci] ([99e9112](https://github.com/w-disaster/transcendent-multiclass/commit/99e91127ac704ba1d76e7bcd6958370ed5af04ef))
* **release:** 1.0.0 [skip ci] ([c519bb0](https://github.com/w-disaster/transcendent-multiclass/commit/c519bb057b5219cfb7130ad5df95e8a22a3dab67))
* remove tqdm in rf ncm ([efe466d](https://github.com/w-disaster/transcendent-multiclass/commit/efe466d152e3e873dfcd47beb08e6f75b84c002f))
* remove unused files ([0342fc7](https://github.com/w-disaster/transcendent-multiclass/commit/0342fc798de586e04c1504b8c78b89245194d33a))
* remove useless files ([28b04ca](https://github.com/w-disaster/transcendent-multiclass/commit/28b04cada7f13a92216a32f10bb837ffb924318d))
* remove useless files, update gitignore ([7055fe1](https://github.com/w-disaster/transcendent-multiclass/commit/7055fe1c9d30e98ceeda38c715190b43dd624fcd))
* rename package to transcendent ([471cd0f](https://github.com/w-disaster/transcendent-multiclass/commit/471cd0f3c650b18cbe580e024655e977144d8d17))
* restore malware dataset ([c1e29eb](https://github.com/w-disaster/transcendent-multiclass/commit/c1e29eb69dbe7c5f64898d6a5fb2e67bae68bfe3))
* restore run script ([dbfac36](https://github.com/w-disaster/transcendent-multiclass/commit/dbfac36d6ff5b1945a053f480932002036b059f6))
* rf-10-knn to rf-5-knn ([09031b7](https://github.com/w-disaster/transcendent-multiclass/commit/09031b7944c7f827c280a3cc5f27559f13b81d38))
* rm changelog ([787df4d](https://github.com/w-disaster/transcendent-multiclass/commit/787df4db92e90872d983ad3e98b71964f144fc7b))
* rm Changelog ([33637ea](https://github.com/w-disaster/transcendent-multiclass/commit/33637ea07382737409fc5ce1c1bb2fbb3099136c))
* update concept drift notebook ([39ea4f8](https://github.com/w-disaster/transcendent-multiclass/commit/39ea4f8a0f0965ad88a20799aa256c84552155ea))
* update dataset loading ([9d6d5a3](https://github.com/w-disaster/transcendent-multiclass/commit/9d6d5a39c4567e04f6ba8f5a80e43af4d80cd7f7))
* update gitignore ([707d58d](https://github.com/w-disaster/transcendent-multiclass/commit/707d58d96de4cbadd463163db6166997ea518d50))
* update notebook with correct conf score ([c48a1d9](https://github.com/w-disaster/transcendent-multiclass/commit/c48a1d98b6d303ec0b564de6081700d39c0397b4))
* use all labels in ice for testing stats ([f7f35ee](https://github.com/w-disaster/transcendent-multiclass/commit/f7f35eeff186f7d64a3416948ce6405424641486))
* use malware dataset ([5c5b42d](https://github.com/w-disaster/transcendent-multiclass/commit/5c5b42d2579c39eb1e45e3ad3679686ce398e26b))
* use micro as scores average ([561e0aa](https://github.com/w-disaster/transcendent-multiclass/commit/561e0aa3cddaa3809448e11bdf0d038c4f6b0c2d))
* use random forest classifier instead of svm, ncm based on rf proximities ([82a4b16](https://github.com/w-disaster/transcendent-multiclass/commit/82a4b166b5bfe90158e27b32aeb5766afe228511))

### Style improvements

* format code ([84bdafd](https://github.com/w-disaster/transcendent-multiclass/commit/84bdafdb9e4109aeb9968af0f2fd53dfef4fa706))
* format code ([bb9433b](https://github.com/w-disaster/transcendent-multiclass/commit/bb9433bf67888fbd524e06525ea043604fba376e))

### Refactoring

* remove data split analysis dir ([a243502](https://github.com/w-disaster/transcendent-multiclass/commit/a24350251c883363473c0da678e5e27f050adada))
* remove useless file ([70c28df](https://github.com/w-disaster/transcendent-multiclass/commit/70c28dfbd1cb8cbd7e322025afcf737f7121bbc8))
* remove useless files ([ad1c78a](https://github.com/w-disaster/transcendent-multiclass/commit/ad1c78ae07cfddf8fcc26444a0996757bc4805bd))
