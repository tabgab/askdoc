askDOC is meant to be an easy way to check the OMNEST / OMNeT++ documentation, and get answers to your question.

The tool will work with an embedding database, the idea is to have that databas downloaded or created only once locally
and to work from there to preserve resources.

You will still have to provide your unique and personal OPENAI KEY to access these functions, by pasting it into the configuration file.

This tool does not replace the documentation, and may at time give wrong answers so use common sense and verify the answers you get. 
You use this tool entirely at your own responsibility, no guarantees are given what so ever.

BEFORE YOU FIRST RUN THIS TOOL:
- Open openai_key.txt and paste your own unique openAI key into this file. Remove the placeholder text. The only thing in this text file should be your OpenAI key.

- Either copy the OMNEST/OMNeT++ documentation pdf files into this folder, or make sure you first run the tool specifyin where to find them.
For example:
    > python askDOC -q "How can I colorize an icon?" -pdfp "../omnetpp-6.0.1/doc/"