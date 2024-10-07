from database.recommendation_system_database import db
from datetime import datetime

class UserBrowsingHistory(db.Model):
    __tablename__ = 'user_browsing_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    asin = db.Column(db.String(50), nullable=False)
    viewed_at = db.Column(db.DateTime, default=datetime.utcnow)
