.PHONY: test build

DOCKER = docker
DOCKER_JDK_IMAGE = openjdk:7
CONTAINER_NAME = xgboost-predictor-builder

# https://github.com/docker-library/openjdk/issues/117#issuecomment-307222367
workaround_sslexception = sed -i -e 's/^jdk.certpath.disabledAlgorithms=/jdk.certpath.disabledAlgorithms=ECDSA, /' /usr/lib/jvm/java-7-openjdk-amd64/jre/lib/security/java.security

test:
	$(DOCKER) run --rm -it \
		--name $(CONTAINER_NAME) \
		-v ~/.gradle:/root/.gradle \
		-v $(PWD):/work \
		-w /work \
		$(DOCKER_JDK_IMAGE) \
		/bin/bash -c "$(workaround_sslexception) && ./gradlew clean test --no-daemon"

build:
	$(DOCKER) run --rm -it \
		--name $(CONTAINER_NAME) \
		-v ~/.gradle:/root/.gradle \
		-v $(PWD):/work \
		-w /work \
		$(DOCKER_JDK_IMAGE) \
		/bin/bash -c "$(workaround_sslexception) && ./gradlew clean build --no-daemon"
