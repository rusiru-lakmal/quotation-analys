import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { Document, Schema as MongooseSchema } from 'mongoose';

@Schema({ timestamps: true })
export class Quotation extends Document {
  @Prop({ required: true })
  originalFile: string;

  @Prop({ type: MongooseSchema.Types.Mixed, required: true })
  extractedData: any;

  @Prop({ default: 'PENDING_REVIEW' })
  status: string;
}

export const QuotationSchema = SchemaFactory.createForClass(Quotation);
