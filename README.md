# semantic_analysis_essential
A project on Natural Language Inference, focussing on detecting contradictions in German text. The main data set used is a machine-translated version of SNLI (Stanford Natural Language Inference). In addition to that, a data set of 2329 Amazon review pairs is collected and labeled with respect to whether there is a contradiction between the two reviews. Different feature extraction and classification techniques are tested and compared regarding their performance on the original and translated data. We use simple Bag-of-Words based approaches, that can be enhanced with handcrafted semantic and syntactic features, as well as approaches based on Sequence-to-Sequence and End-to-End Recurrent Neural Networks.

Part of this research was published at IEEE CIDM 2019 and can be accessed here: https://www.researchgate.net/publication/338407395_Towards_Contradiction_Detection_in_German_A_Translation-driven_Approach

It was also used for my Master's Thesis with the topic "Neural Network Methods for Natural Language Inference in German".

This repository contains only the essential code for our research.
