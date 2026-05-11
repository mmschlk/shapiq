# MongoDB Storage

This module connects to a MongoDB database and provides functions to store and retrieve data related to the leaderboard, such as run results, approximator types, and ground truth methods. 

It uses the `pymongo` library for database interactions and `dotenv` for managing environment variables (database login credentials). Using `Python 3.12.3`, but code should still be compatible with newer versions of Python.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
``` 

Or 
```bash
pip install numpy dotenv pymongo
```

To use it a valid MongoDB connection string must be provided in the environment variable `MONGODB_URI`. To create such a database one can follow the instructions in the [MongoDB Atlas documentation](https://www.mongodb.com/docs/atlas/getting-started/).

Summary (of the instructions)
1. Create a MongoDB Atlas account and set up a cluster (free tier allows for up to 512 MB of storage).
2. Create a database user with appropriate permissions (first user is the database admin).
3. Whitelist IP address (MongoDB Atlas requires to whitelist the IP address in order to allow connections).

