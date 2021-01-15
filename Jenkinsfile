pipeline {
    agent docker

    stages {
        stage('Build') {
            steps {
	    	sh("make build")
                echo 'Building..'
		sh("make clean")
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
}