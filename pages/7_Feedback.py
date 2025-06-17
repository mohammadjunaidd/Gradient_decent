import streamlit as st
import csv
import re
from textblob import TextBlob

# Add custom CSS for styling
custom_css = """
    <style>
        .header {
            font-size: 2em;
            font-weight: bold;
            color: #5A78F0; /* Bright blue */
            text-align: center;
        }
        .subheader {
            font-size: 1.2em;
            color: #6D6D6D; /* Dark gray */
            text-align: center;
            margin-bottom: 20px;
        }
        .emoji {
            font-size: 1.5em;
            margin-right: 10px;
        }
        input[type="text"], input[type="email"], textarea {
            border-radius: 8px;
            border: 1px solid #ccc;
            padding: 10px;
            font-size: 1em;
        }
        button {
            background: linear-gradient(90deg, #6A5ACD, #7B68EE); /* Purple gradient */
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 1em;
            cursor: pointer;
        }
        button:hover {
            background: linear-gradient(90deg, #5A78F0, #5F87FF); /* Blue gradient on hover */
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Function to save feedback into a CSV file
def save_feedback(name, email, feedback_type, message, rating, sentiment):
    with open(r"D:\Github\Publication\gradient\feedbacks.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, email, feedback_type, message, rating, sentiment])

# Function to validate email
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# Function to analyze sentiment
def analyze_sentiment(feedback):
    analysis = TextBlob(feedback)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# Create Streamlit app
def feedback_form():
    st.markdown('<div class="header">We Appreciate Your Feedback! 🌟</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Your thoughts guide us to improve and innovate! 💡🚀</div>', unsafe_allow_html=True)

    # Collect user input
    name = st.text_input("May I have your name, please? 🙋‍♂️")  # Optional
    email = st.text_input("Email Address (We’ll keep it safe!) 🔐")  # Required
    feedback_type = st.selectbox(
        "How would you categorize your feedback? 🤔",
        ["Compliment 👏", "Idea 💭", "Concern ⚠️", "Other 📝"]
    )
    message = st.text_area("Let us know your thoughts! 🗣️")
    rating = st.slider("Rate your experience with us! 🎯", 1, 5, 3)

    # Submit button
    if st.button("Send Feedback 🚀"):
        if email and message:
            if not is_valid_email(email):
                st.error("Please enter a valid email address.")
                return

            sentiment = analyze_sentiment(message)
            save_feedback(name, email, feedback_type, message, rating, sentiment)

            st.success("Thanks for sharing! You're awesome! 🙌")
            st.markdown(f"**Type of Feedback**: {feedback_type} ✨")
            st.markdown(f"**Sentiment Analysis**: {sentiment} 💬")
            st.markdown(f"**Your Rating**: {rating} ⭐")
        else:
            st.warning("Please provide both email and feedback.")

# Run the feedback form app
if __name__ == "__main__":
    feedback_form()
