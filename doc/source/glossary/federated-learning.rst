What is Federated Learning?
===========================

Federated Learning is a technique to train machine learning algorithms such as deep learning models on datasets which one can, for various reasons, not merge. The first thing we will describe is why we need Federated Learning. Without Federated Learning, a dataset had to be centrally aggregated, which was not always possible due to regulatory or technical reasons. The requirement to aggregate data centrally limited the number of possible use-cases for machine learning drastically. Here are two examples where centralizing data is extremely hard or impossible:

*Training AI models to detect skin cancer*
Training a model which can detect skin cancer requires a lot of labelled patient data. In most countries, sharing and using medical data is challenging due to various regulations, which makes aggregating a meaningful amount of data near impossible.

*User Recommendation Systems*
When training a recommender AI model, it is essential to have the correct data available. Aggregating all user data can violate various data privacy regulations, varying significantly from country to country. Given these challenges training good recommender AI models can become challenging.

How Federate Learning works is the next thing we are going to describe. When training an AI model federated, the training does not happen centralized. We will describe what is happening in the most basic form of Federated Learning.

*Step 1*
Instead of aggregating the data, a coordinator, in our case, the Flower Server, sends a global model to all locations where data resides. Each of our data silos or Flower clients (e.g. a smartphone) has to be able to train an AI model for a fraction of the time an AI model is usually training to refine and improve it. After producing an update, the Flower client sends the refined model back to the server.

*Step 2*
After receiving at least two updates, the Flower server will aggregate all model updates into a new global model. There are various strategies how to do that available in Flower. After the new global model is available, the process starts anew with Steps 1 and 2 repeated until the training performance is satisfying.
