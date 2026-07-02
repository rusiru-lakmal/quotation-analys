import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { QuotationModule } from './quotation/quotation.module';

@Module({
  imports: [
    MongooseModule.forRoot('mongodb://admin:supersecretpassword@localhost:27017/quotation_ai?authSource=admin'),
    QuotationModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
