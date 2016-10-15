# WhatYouSee

 This project contains the nodejs backend and cordova projeect of an app that allows the user to receive a simple audio description of pictures taken with a smartphone camera. The app uses two Watson cognitive services available at IBM Bluemix .  The Visual Recognition service is used to analyze the pictures and provide a list of labels that describe the picture with a certain confidence (the label score). The list of labels is converted to a descriptive sentence and the Text to Speech service is used to convert the sentence into an audio file. The audio is then executed by the smartphone.

The mobile app is build with Cordova and JQuery Mobile and the backend that establishes the communication between the mobile device and the cognitive services is build with NodeJS.

## Getting Started

1. Create a Bluemix Account

  [Sign up][sign_up] in Bluemix, or use an existing account. Watson Services in Beta are free to use.

2. Download and install the [Cloud-foundry CLI][cloud_foundry] tool

3. Edit the `manifest.yml` file and change the `<application-name>` to something unique.
  ```none
applications:
- services
  - visual-recognition-service
  - text-to-speech-service
  name: <application-name>
  command: node app.js
  path: .
  memory: 128M
  ```
  The name you use will determinate your application url initially, e.g. `<application-name>.mybluemix.net`.

4. Connect to Bluemix in the command line tool
  ```sh
  $ cf api https://api.ng.bluemix.net
  $ cf login -u <your user ID>
  ```

5. Create the Visual Recognition service in Bluemix
  ```sh
  $ cf create-service visual_recognition free visual-recognition-service
  ```
6. Create the Text To Speech service in Bluemix
  ```sh
  $ cf create-service text_to_speech standard text-to-speech-service
  ```

7. Push it live!
  ```sh
  $ cf push
  ```

See the full [Getting Started][getting_started] documentation for more details, including code snippets and references.

## Running locally
  The application uses [Node.js](http://nodejs.org) and [npm](https://www.npmjs.com) so you will have to download and install them as part of the steps below.

1. Copy the credentials from your `visual-recognition-service` and `text-to-speech-service` services in Bluemix to `app.js`, you can see the credentials using:

    ```sh
    $ cf env <application-name>
    ```
    Example output:
    ```sh
    System-Provided:
    {
    "VCAP_SERVICES": {
      "visual_recognition": [{
          "credentials": {
            "url": "<url>",
            "password": "<password>",
            "username": "<username>"
          },
        "label": "visual_recognition",
        "name": "visual-recognition-service",
        "plan": "free"
     }],
      "text_to_speech": [{
          "credentials": {
            "url": "<url>",
            "password": "<password>",
            "username": "<username>"
          },
        "label": "text_to_speech",
        "name": "text-to-speech-service",
        "plan": "standard"}]
      }
    }
    ```

    You need to copy `username`, `password` and `url`.

2. Install [Node.js](http://nodejs.org/)
3. Go to the project folder in a terminal and run:
    `npm install`
4. Start the application
5.  `node app.js`
6. Go to `http://localhost:3000`

## Build and Run the Cordova app

1. Install [Apache Cordova](https://cordova.apache.org/)

2. Navigate to whatyousee-cordova folder

3. Run `cordova prepare` from whatyousee-cordova  folder

4. Connect an Android device to the computer USB port and run `cordova run android --device`

Turn up the volume of your device while running this app


## Troubleshooting 

To troubleshoot your Bluemix app the main useful source of information are the logs, to see them, run:

  ```sh
  $ cf logs <application-name> --recent
  ```

## License

  This sample code is licensed under Apache 2.0. Full license text is available in [LICENSE](License.txt).